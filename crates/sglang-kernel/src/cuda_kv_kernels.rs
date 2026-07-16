use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_KV_COPY_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void sglang_scatter_kv_pair_bytes(
    const unsigned char* keys,
    const unsigned char* values,
    const unsigned long long* slots,
    unsigned char* pool,
    unsigned int* error_flag,
    unsigned long long row_count,
    unsigned long long slot_count,
    unsigned long long page_size,
    unsigned long long page_stride_bytes,
    unsigned long long key_in_page_offset,
    unsigned long long value_in_page_offset,
    unsigned long long row_bytes,
    unsigned long long key_row_stride_bytes,
    unsigned long long value_row_stride_bytes) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long plane_bytes = row_count * row_bytes;
  if (index >= plane_bytes * 2) {
    return;
  }

  const bool is_value = index >= plane_bytes;
  const unsigned long long plane_index = is_value ? index - plane_bytes : index;
  const unsigned long long row = plane_index / row_bytes;
  const unsigned long long byte = plane_index % row_bytes;
  const unsigned long long slot = slots[row];
  if (slot >= slot_count) {
    atomicExch(error_flag, 1u);
    return;
  }

  const unsigned long long page = slot / page_size;
  const unsigned long long token = slot % page_size;
  const unsigned long long in_page_offset =
      is_value ? value_in_page_offset : key_in_page_offset;
  const unsigned long long destination =
      page * page_stride_bytes + in_page_offset + token * row_bytes + byte;
  const unsigned long long source =
      row * (is_value ? value_row_stride_bytes : key_row_stride_bytes) + byte;
  pool[destination] = is_value ? values[source] : keys[source];
}

extern "C" __global__ void sglang_gather_kv_pair_bytes(
    const unsigned char* pool,
    const unsigned long long* slots,
    unsigned char* keys,
    unsigned char* values,
    unsigned int* error_flag,
    unsigned long long row_count,
    unsigned long long slot_count,
    unsigned long long page_size,
    unsigned long long page_stride_bytes,
    unsigned long long key_in_page_offset,
    unsigned long long value_in_page_offset,
    unsigned long long row_bytes,
    unsigned long long key_row_stride_bytes,
    unsigned long long value_row_stride_bytes) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long plane_bytes = row_count * row_bytes;
  if (index >= plane_bytes * 2) {
    return;
  }

  const bool is_value = index >= plane_bytes;
  const unsigned long long plane_index = is_value ? index - plane_bytes : index;
  const unsigned long long row = plane_index / row_bytes;
  const unsigned long long byte = plane_index % row_bytes;
  const unsigned long long slot = slots[row];
  if (slot >= slot_count) {
    atomicExch(error_flag, 1u);
    return;
  }

  const unsigned long long page = slot / page_size;
  const unsigned long long token = slot % page_size;
  const unsigned long long in_page_offset =
      is_value ? value_in_page_offset : key_in_page_offset;
  const unsigned long long source =
      page * page_stride_bytes + in_page_offset + token * row_bytes + byte;
  const unsigned long long destination =
      row * (is_value ? value_row_stride_bytes : key_row_stride_bytes) + byte;
  if (is_value) {
    values[destination] = pool[source];
  } else {
    keys[destination] = pool[source];
  }
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;
const SLOT_INDEX_BYTES: usize = size_of::<u64>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKvPairCopyLayout {
    page_size: usize,
    page_stride_bytes: usize,
    key_in_page_offset: usize,
    value_in_page_offset: usize,
    row_bytes: usize,
}

impl CudaKvPairCopyLayout {
    pub const fn new(
        page_size: usize,
        page_stride_bytes: usize,
        key_in_page_offset: usize,
        value_in_page_offset: usize,
        row_bytes: usize,
    ) -> Self {
        Self {
            page_size,
            page_stride_bytes,
            key_in_page_offset,
            value_in_page_offset,
            row_bytes,
        }
    }

    pub const fn page_size(self) -> usize {
        self.page_size
    }

    pub const fn page_stride_bytes(self) -> usize {
        self.page_stride_bytes
    }

    pub const fn key_in_page_offset(self) -> usize {
        self.key_in_page_offset
    }

    pub const fn value_in_page_offset(self) -> usize {
        self.value_in_page_offset
    }

    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKvPairCopyPlan {
    row_count: usize,
    slot_count: usize,
    layout: CudaKvPairCopyLayout,
    key_row_stride_bytes: usize,
    value_row_stride_bytes: usize,
    key_rows_required_bytes: usize,
    value_rows_required_bytes: usize,
    slot_indices_required_bytes: usize,
    pool_required_bytes: usize,
    grid: CudaLaunchDimensions,
}

impl CudaKvPairCopyPlan {
    pub fn new(
        row_count: usize,
        slot_count: usize,
        layout: CudaKvPairCopyLayout,
        key_row_stride_bytes: usize,
        value_row_stride_bytes: usize,
    ) -> Result<Self, CudaKvPairCopyError> {
        for (dimension, value) in [
            ("row_count", row_count),
            ("slot_count", slot_count),
            ("page_size", layout.page_size),
            ("page_stride_bytes", layout.page_stride_bytes),
            ("row_bytes", layout.row_bytes),
        ] {
            if value == 0 {
                return Err(CudaKvPairCopyError::ZeroDimension(dimension));
            }
        }
        validate_row_stride("key", layout.row_bytes, key_row_stride_bytes)?;
        validate_row_stride("value", layout.row_bytes, value_row_stride_bytes)?;

        let tensor_page_bytes = layout
            .page_size
            .checked_mul(layout.row_bytes)
            .ok_or(CudaKvPairCopyError::ShapeOverflow)?;
        let key_page_end = validate_tensor_page_region(
            "key",
            layout.key_in_page_offset,
            tensor_page_bytes,
            layout.page_stride_bytes,
        )?;
        let value_page_end = validate_tensor_page_region(
            "value",
            layout.value_in_page_offset,
            tensor_page_bytes,
            layout.page_stride_bytes,
        )?;
        if layout.key_in_page_offset < value_page_end && layout.value_in_page_offset < key_page_end
        {
            return Err(CudaKvPairCopyError::TensorPageRegionsOverlap {
                key_offset: layout.key_in_page_offset,
                value_offset: layout.value_in_page_offset,
                tensor_page_bytes,
            });
        }

        let key_rows_required_bytes =
            strided_rows_required_bytes(row_count, key_row_stride_bytes, layout.row_bytes)?;
        let value_rows_required_bytes =
            strided_rows_required_bytes(row_count, value_row_stride_bytes, layout.row_bytes)?;
        let slot_indices_required_bytes = row_count
            .checked_mul(SLOT_INDEX_BYTES)
            .ok_or(CudaKvPairCopyError::ShapeOverflow)?;
        let last_slot = slot_count - 1;
        let last_page = last_slot / layout.page_size;
        let last_token = last_slot % layout.page_size;
        let last_tensor_end = key_page_end.max(value_page_end);
        let pool_required_bytes = last_page
            .checked_mul(layout.page_stride_bytes)
            .and_then(|offset| {
                offset.checked_add(
                    last_tensor_end
                        .checked_sub(tensor_page_bytes)?
                        .checked_add(last_token.checked_mul(layout.row_bytes)?)?
                        .checked_add(layout.row_bytes)?,
                )
            })
            .ok_or(CudaKvPairCopyError::ShapeOverflow)?;

        let thread_count = row_count
            .checked_mul(layout.row_bytes)
            .and_then(|value| value.checked_mul(2))
            .ok_or(CudaKvPairCopyError::ShapeOverflow)?;
        let block_size = CUDA_BLOCK_SIZE as usize;
        let block_count = thread_count
            .checked_add(block_size - 1)
            .ok_or(CudaKvPairCopyError::ShapeOverflow)?
            / block_size;
        let grid_x =
            u32::try_from(block_count).map_err(|_| CudaKvPairCopyError::DimensionTooLarge {
                dimension: "grid blocks",
                value: block_count,
                maximum: u32::MAX as usize,
            })?;

        Ok(Self {
            row_count,
            slot_count,
            layout,
            key_row_stride_bytes,
            value_row_stride_bytes,
            key_rows_required_bytes,
            value_rows_required_bytes,
            slot_indices_required_bytes,
            pool_required_bytes,
            grid: CudaLaunchDimensions::new(grid_x, 1, 1),
        })
    }

    pub const fn row_count(self) -> usize {
        self.row_count
    }

    pub const fn slot_count(self) -> usize {
        self.slot_count
    }

    pub const fn layout(self) -> CudaKvPairCopyLayout {
        self.layout
    }

    pub const fn key_row_stride_bytes(self) -> usize {
        self.key_row_stride_bytes
    }

    pub const fn value_row_stride_bytes(self) -> usize {
        self.value_row_stride_bytes
    }

    pub const fn key_rows_required_bytes(self) -> usize {
        self.key_rows_required_bytes
    }

    pub const fn value_rows_required_bytes(self) -> usize {
        self.value_rows_required_bytes
    }

    pub const fn slot_indices_required_bytes(self) -> usize {
        self.slot_indices_required_bytes
    }

    pub const fn pool_required_bytes(self) -> usize {
        self.pool_required_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaKvPairCopyError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    ZeroDimension(&'static str),
    ShapeOverflow,
    DimensionTooLarge {
        dimension: &'static str,
        value: usize,
        maximum: usize,
    },
    RowStrideTooSmall {
        tensor: &'static str,
        row_bytes: usize,
        row_stride_bytes: usize,
    },
    TensorPageRegionOutOfBounds {
        tensor: &'static str,
        offset: usize,
        tensor_page_bytes: usize,
        page_stride_bytes: usize,
    },
    TensorPageRegionsOverlap {
        key_offset: usize,
        value_offset: usize,
        tensor_page_bytes: usize,
    },
    DeviceMismatch {
        allocation: &'static str,
        expected_ordinal: usize,
        actual_ordinal: usize,
    },
    DeviceSlotIndexOutOfRange {
        slot_count: usize,
    },
}

impl fmt::Display for CudaKvPairCopyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA KV copy operation failed: {error}"),
            Self::Nvrtc(error) => write!(formatter, "CUDA KV copy compilation failed: {error}"),
            Self::ZeroDimension(dimension) => {
                write!(
                    formatter,
                    "CUDA KV copy dimension {dimension} must be non-zero"
                )
            }
            Self::ShapeOverflow => formatter.write_str("CUDA KV copy tensor shape overflowed"),
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA KV copy dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::RowStrideTooSmall {
                tensor,
                row_bytes,
                row_stride_bytes,
            } => write!(
                formatter,
                "CUDA KV copy {tensor} row stride {row_stride_bytes} bytes is smaller than row width {row_bytes} bytes"
            ),
            Self::TensorPageRegionOutOfBounds {
                tensor,
                offset,
                tensor_page_bytes,
                page_stride_bytes,
            } => write!(
                formatter,
                "CUDA KV copy {tensor} page region [{offset}, {}) exceeds page stride {page_stride_bytes}",
                offset.saturating_add(*tensor_page_bytes)
            ),
            Self::TensorPageRegionsOverlap {
                key_offset,
                value_offset,
                tensor_page_bytes,
            } => write!(
                formatter,
                "CUDA KV copy key/value page regions overlap: key offset {key_offset}, value offset {value_offset}, tensor page size {tensor_page_bytes} bytes"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA KV copy allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
            Self::DeviceSlotIndexOutOfRange { slot_count } => write!(
                formatter,
                "CUDA KV copy device slot map contains an index outside {slot_count} slots"
            ),
        }
    }
}

impl std::error::Error for CudaKvPairCopyError {}

impl From<CudaError> for CudaKvPairCopyError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaKvPairCopyError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaKvPairCopyKernels {
    context: CudaContext,
    _module: CudaModule,
    scatter: CudaFunction,
    gather: CudaFunction,
    error_flag: CudaDeviceAllocation,
}

impl CudaKvPairCopyKernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaKvPairCopyError> {
        let compiler = NvrtcCompiler::load()?;
        let ptx = compiler.compile_ptx(
            CUDA_KV_COPY_KERNEL_SOURCE,
            "sglang_kv_copy_kernels.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        let scatter = module.get_function("sglang_scatter_kv_pair_bytes")?;
        let gather = module.get_function("sglang_gather_kv_pair_bytes")?;
        let mut error_flag = context.allocate(size_of::<u32>())?;
        error_flag.fill(0)?;
        Ok(Self {
            context: context.clone(),
            _module: module,
            scatter,
            gather,
            error_flag,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scatter(
        &mut self,
        plan: CudaKvPairCopyPlan,
        keys: &CudaDeviceAllocation,
        keys_offset: usize,
        values: &CudaDeviceAllocation,
        values_offset: usize,
        slot_indices: &CudaDeviceAllocation,
        slot_indices_offset: usize,
        pool: &mut CudaDeviceAllocation,
        pool_offset: usize,
    ) -> Result<(), CudaKvPairCopyError> {
        self.validate_device("keys", keys)?;
        self.validate_device("values", values)?;
        self.validate_device("slot_indices", slot_indices)?;
        self.validate_device("pool", pool)?;

        let mut keys_ptr = keys.device_ptr_at(keys_offset, plan.key_rows_required_bytes)?;
        let mut values_ptr = values.device_ptr_at(values_offset, plan.value_rows_required_bytes)?;
        let mut slots_ptr =
            slot_indices.device_ptr_at(slot_indices_offset, plan.slot_indices_required_bytes)?;
        let mut pool_ptr = pool.device_ptr_at(pool_offset, plan.pool_required_bytes)?;
        self.launch(
            CopyDirection::Scatter,
            plan,
            &mut keys_ptr,
            &mut values_ptr,
            &mut slots_ptr,
            &mut pool_ptr,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather(
        &mut self,
        plan: CudaKvPairCopyPlan,
        pool: &CudaDeviceAllocation,
        pool_offset: usize,
        slot_indices: &CudaDeviceAllocation,
        slot_indices_offset: usize,
        keys: &mut CudaDeviceAllocation,
        keys_offset: usize,
        values: &mut CudaDeviceAllocation,
        values_offset: usize,
    ) -> Result<(), CudaKvPairCopyError> {
        self.validate_device("pool", pool)?;
        self.validate_device("slot_indices", slot_indices)?;
        self.validate_device("keys", keys)?;
        self.validate_device("values", values)?;

        let mut pool_ptr = pool.device_ptr_at(pool_offset, plan.pool_required_bytes)?;
        let mut slots_ptr =
            slot_indices.device_ptr_at(slot_indices_offset, plan.slot_indices_required_bytes)?;
        let mut keys_ptr = keys.device_ptr_at(keys_offset, plan.key_rows_required_bytes)?;
        let mut values_ptr = values.device_ptr_at(values_offset, plan.value_rows_required_bytes)?;
        self.launch(
            CopyDirection::Gather,
            plan,
            &mut keys_ptr,
            &mut values_ptr,
            &mut slots_ptr,
            &mut pool_ptr,
        )
    }

    fn launch(
        &mut self,
        direction: CopyDirection,
        plan: CudaKvPairCopyPlan,
        keys_ptr: &mut u64,
        values_ptr: &mut u64,
        slots_ptr: &mut u64,
        pool_ptr: &mut u64,
    ) -> Result<(), CudaKvPairCopyError> {
        self.error_flag.fill(0)?;
        let mut error_ptr = self.error_flag.device_ptr_at(0, size_of::<u32>())?;
        let mut row_count = dimension_u64("row_count", plan.row_count)?;
        let mut slot_count = dimension_u64("slot_count", plan.slot_count)?;
        let mut page_size = dimension_u64("page_size", plan.layout.page_size)?;
        let mut page_stride_bytes =
            dimension_u64("page_stride_bytes", plan.layout.page_stride_bytes)?;
        let mut key_in_page_offset =
            dimension_u64("key_in_page_offset", plan.layout.key_in_page_offset)?;
        let mut value_in_page_offset =
            dimension_u64("value_in_page_offset", plan.layout.value_in_page_offset)?;
        let mut row_bytes = dimension_u64("row_bytes", plan.layout.row_bytes)?;
        let mut key_row_stride_bytes =
            dimension_u64("key_row_stride_bytes", plan.key_row_stride_bytes)?;
        let mut value_row_stride_bytes =
            dimension_u64("value_row_stride_bytes", plan.value_row_stride_bytes)?;

        let function = match direction {
            CopyDirection::Scatter => &self.scatter,
            CopyDirection::Gather => &self.gather,
        };
        let mut arguments = match direction {
            CopyDirection::Scatter => [
                argument_pointer(keys_ptr),
                argument_pointer(values_ptr),
                argument_pointer(slots_ptr),
                argument_pointer(pool_ptr),
                argument_pointer(&mut error_ptr),
                argument_pointer(&mut row_count),
                argument_pointer(&mut slot_count),
                argument_pointer(&mut page_size),
                argument_pointer(&mut page_stride_bytes),
                argument_pointer(&mut key_in_page_offset),
                argument_pointer(&mut value_in_page_offset),
                argument_pointer(&mut row_bytes),
                argument_pointer(&mut key_row_stride_bytes),
                argument_pointer(&mut value_row_stride_bytes),
            ],
            CopyDirection::Gather => [
                argument_pointer(pool_ptr),
                argument_pointer(slots_ptr),
                argument_pointer(keys_ptr),
                argument_pointer(values_ptr),
                argument_pointer(&mut error_ptr),
                argument_pointer(&mut row_count),
                argument_pointer(&mut slot_count),
                argument_pointer(&mut page_size),
                argument_pointer(&mut page_stride_bytes),
                argument_pointer(&mut key_in_page_offset),
                argument_pointer(&mut value_in_page_offset),
                argument_pointer(&mut row_bytes),
                argument_pointer(&mut key_row_stride_bytes),
                argument_pointer(&mut value_row_stride_bytes),
            ],
        };
        unsafe {
            function.launch(
                plan.grid,
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                0,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;

        let mut error_flag = [0_u8; size_of::<u32>()];
        self.error_flag.copy_to_host(0, &mut error_flag)?;
        if u32::from_ne_bytes(error_flag) != 0 {
            return Err(CudaKvPairCopyError::DeviceSlotIndexOutOfRange {
                slot_count: plan.slot_count,
            });
        }
        Ok(())
    }

    fn validate_device(
        &self,
        allocation_name: &'static str,
        allocation: &CudaDeviceAllocation,
    ) -> Result<(), CudaKvPairCopyError> {
        let expected_ordinal = self.context.device_ordinal();
        let actual_ordinal = allocation.device_ordinal();
        if actual_ordinal == expected_ordinal {
            Ok(())
        } else {
            Err(CudaKvPairCopyError::DeviceMismatch {
                allocation: allocation_name,
                expected_ordinal,
                actual_ordinal,
            })
        }
    }
}

#[derive(Clone, Copy)]
enum CopyDirection {
    Scatter,
    Gather,
}

fn validate_row_stride(
    tensor: &'static str,
    row_bytes: usize,
    row_stride_bytes: usize,
) -> Result<(), CudaKvPairCopyError> {
    if row_stride_bytes < row_bytes {
        Err(CudaKvPairCopyError::RowStrideTooSmall {
            tensor,
            row_bytes,
            row_stride_bytes,
        })
    } else {
        Ok(())
    }
}

fn validate_tensor_page_region(
    tensor: &'static str,
    offset: usize,
    tensor_page_bytes: usize,
    page_stride_bytes: usize,
) -> Result<usize, CudaKvPairCopyError> {
    let end = offset
        .checked_add(tensor_page_bytes)
        .ok_or(CudaKvPairCopyError::ShapeOverflow)?;
    if end > page_stride_bytes {
        Err(CudaKvPairCopyError::TensorPageRegionOutOfBounds {
            tensor,
            offset,
            tensor_page_bytes,
            page_stride_bytes,
        })
    } else {
        Ok(end)
    }
}

fn strided_rows_required_bytes(
    row_count: usize,
    row_stride_bytes: usize,
    row_bytes: usize,
) -> Result<usize, CudaKvPairCopyError> {
    (row_count - 1)
        .checked_mul(row_stride_bytes)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or(CudaKvPairCopyError::ShapeOverflow)
}

fn dimension_u64(dimension: &'static str, value: usize) -> Result<u64, CudaKvPairCopyError> {
    u64::try_from(value).map_err(|_| CudaKvPairCopyError::DimensionTooLarge {
        dimension,
        value,
        maximum: u64::MAX as usize,
    })
}

fn argument_pointer<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_layout() -> CudaKvPairCopyLayout {
        CudaKvPairCopyLayout::new(4, 512, 256, 384, 32)
    }

    #[test]
    fn plan_covers_strided_sources_and_last_physical_slot() {
        let plan = CudaKvPairCopyPlan::new(3, 12, test_layout(), 40, 48)
            .expect("valid KV copy plan should build");

        assert_eq!(plan.row_count(), 3);
        assert_eq!(plan.slot_count(), 12);
        assert_eq!(plan.layout(), test_layout());
        assert_eq!(plan.key_rows_required_bytes(), 112);
        assert_eq!(plan.value_rows_required_bytes(), 128);
        assert_eq!(plan.slot_indices_required_bytes(), 24);
        assert_eq!(plan.pool_required_bytes(), 1_536);
        assert_eq!(plan.grid, CudaLaunchDimensions::new(1, 1, 1));
    }

    #[test]
    fn plan_rejects_invalid_strides_and_page_regions() {
        assert_eq!(
            CudaKvPairCopyPlan::new(1, 4, test_layout(), 31, 32),
            Err(CudaKvPairCopyError::RowStrideTooSmall {
                tensor: "key",
                row_bytes: 32,
                row_stride_bytes: 31,
            })
        );
        assert_eq!(
            CudaKvPairCopyPlan::new(
                1,
                4,
                CudaKvPairCopyLayout::new(4, 512, 384, 400, 32),
                32,
                32,
            ),
            Err(CudaKvPairCopyError::TensorPageRegionOutOfBounds {
                tensor: "value",
                offset: 400,
                tensor_page_bytes: 128,
                page_stride_bytes: 512,
            })
        );
        assert_eq!(
            CudaKvPairCopyPlan::new(
                1,
                4,
                CudaKvPairCopyLayout::new(4, 512, 256, 320, 32),
                32,
                32,
            ),
            Err(CudaKvPairCopyError::TensorPageRegionsOverlap {
                key_offset: 256,
                value_offset: 320,
                tensor_page_bytes: 128,
            })
        );
    }

    #[test]
    fn plan_rejects_zero_and_overflowing_dimensions() {
        assert_eq!(
            CudaKvPairCopyPlan::new(0, 12, test_layout(), 32, 32),
            Err(CudaKvPairCopyError::ZeroDimension("row_count"))
        );
        assert_eq!(
            CudaKvPairCopyPlan::new(
                usize::MAX,
                12,
                CudaKvPairCopyLayout::new(4, 512, 256, 384, usize::MAX),
                usize::MAX,
                usize::MAX,
            ),
            Err(CudaKvPairCopyError::ShapeOverflow)
        );
    }

    #[test]
    fn embedded_source_exports_dtype_independent_kv_entry_points() {
        assert!(CUDA_KV_COPY_KERNEL_SOURCE.contains("sglang_scatter_kv_pair_bytes"));
        assert!(CUDA_KV_COPY_KERNEL_SOURCE.contains("sglang_gather_kv_pair_bytes"));
        assert!(CUDA_KV_COPY_KERNEL_SOURCE.contains("unsigned char"));
        assert!(CUDA_KV_COPY_KERNEL_SOURCE.contains("atomicExch"));
    }
}
