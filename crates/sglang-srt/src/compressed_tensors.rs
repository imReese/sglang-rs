use std::fmt;

use crate::model_artifacts::{
    LocalModelArtifacts, ModelArtifactError, SafetensorsTensorData, SafetensorsTensorDecodeError,
};

const INT4_PACKED_FACTOR: usize = 8;

#[derive(Debug)]
pub(crate) struct CompressedTensorsInt4Weight {
    rows: usize,
    columns: usize,
    group_size: usize,
    packed: Vec<i32>,
    scales: Vec<f32>,
}

impl CompressedTensorsInt4Weight {
    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        logical_weight_name: &str,
        rows: usize,
        columns: usize,
        group_size: usize,
    ) -> Result<Self, CompressedTensorsError> {
        if rows == 0 || columns == 0 || group_size == 0 {
            return Err(CompressedTensorsError::Invalid(format!(
                "compressed INT4 tensor {logical_weight_name} dimensions and group size must be non-zero"
            )));
        }
        if !columns.is_multiple_of(INT4_PACKED_FACTOR) || !columns.is_multiple_of(group_size) {
            return Err(CompressedTensorsError::Invalid(format!(
                "compressed INT4 tensor {logical_weight_name} input width {columns} must be divisible by packed factor {INT4_PACKED_FACTOR} and group size {group_size}"
            )));
        }
        let base = logical_weight_name.strip_suffix(".weight").ok_or_else(|| {
            CompressedTensorsError::Invalid(format!(
                "compressed INT4 logical weight name {logical_weight_name} must end in .weight"
            ))
        })?;
        let packed_name = format!("{base}.weight_packed");
        let scale_name = format!("{base}.weight_scale");
        let shape_name = format!("{base}.weight_shape");
        let packed = read_required(artifacts, &packed_name)?;
        let scales = read_required(artifacts, &scale_name)?;
        let shape = read_required(artifacts, &shape_name)?;
        validate_shape(&packed, &packed_name, &[rows, columns / INT4_PACKED_FACTOR])?;
        validate_shape(&scales, &scale_name, &[rows, columns / group_size])?;
        validate_shape(&shape, &shape_name, &[2])?;
        if packed.metadata.dtype != "I32" {
            return Err(CompressedTensorsError::Invalid(format!(
                "compressed INT4 tensor {packed_name} must use safetensors I32, found {}",
                packed.metadata.dtype
            )));
        }
        if !matches!(scales.metadata.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(CompressedTensorsError::Invalid(format!(
                "compressed INT4 tensor {scale_name} must use F32, F16, or BF16 scales, found {}",
                scales.metadata.dtype
            )));
        }
        let original_shape = decode_shape(&shape, &shape_name)?;
        if original_shape != [rows, columns] {
            return Err(CompressedTensorsError::Invalid(format!(
                "compressed INT4 tensor {shape_name} declares original shape {original_shape:?}, expected [{rows}, {columns}]"
            )));
        }
        Ok(Self {
            rows,
            columns,
            group_size,
            packed: decode_i32(&packed, &packed_name)?,
            scales: scales.decode_f32_values()?,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn group_size(&self) -> usize {
        self.group_size
    }

    pub(crate) fn packed_i32_bytes(&self) -> Vec<u8> {
        self.packed
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    pub(crate) fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub(crate) fn dequantize_f32(&self) -> Vec<f32> {
        let packed_columns = self.columns / INT4_PACKED_FACTOR;
        let groups_per_row = self.columns / self.group_size;
        let mut values = Vec::with_capacity(self.rows * self.columns);
        for row in 0..self.rows {
            for column in 0..self.columns {
                let word = self.packed[row * packed_columns + column / INT4_PACKED_FACTOR] as u32;
                let nibble = ((word >> ((column % INT4_PACKED_FACTOR) * 4)) & 0x0f) as i32;
                let quantized = nibble - 8;
                let scale = self.scales[row * groups_per_row + column / self.group_size];
                values.push(quantized as f32 * scale);
            }
        }
        values
    }
}

fn read_required(
    artifacts: &LocalModelArtifacts,
    name: &str,
) -> Result<SafetensorsTensorData, CompressedTensorsError> {
    artifacts
        .safetensors()
        .read_tensor(name)?
        .ok_or_else(|| CompressedTensorsError::Invalid(format!("missing tensor {name}")))
}

fn validate_shape(
    tensor: &SafetensorsTensorData,
    name: &str,
    expected: &[usize],
) -> Result<(), CompressedTensorsError> {
    if tensor.metadata.shape != expected {
        return Err(CompressedTensorsError::Invalid(format!(
            "compressed tensor {name} shape {:?} does not match expected {expected:?}",
            tensor.metadata.shape
        )));
    }
    Ok(())
}

fn decode_i32(
    tensor: &SafetensorsTensorData,
    name: &str,
) -> Result<Vec<i32>, CompressedTensorsError> {
    let chunks = tensor.bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(CompressedTensorsError::Invalid(format!(
            "compressed INT4 tensor {name} has invalid byte length {}",
            tensor.bytes.len()
        )));
    }
    Ok(chunks
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_shape(
    tensor: &SafetensorsTensorData,
    name: &str,
) -> Result<[usize; 2], CompressedTensorsError> {
    if tensor.metadata.dtype != "I64" || tensor.bytes.len() != 16 {
        return Err(CompressedTensorsError::Invalid(format!(
            "compressed tensor {name} must contain two I64 dimensions"
        )));
    }
    let mut values = [0_usize; 2];
    for (index, chunk) in tensor.bytes.chunks_exact(8).enumerate() {
        let value = i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        values[index] = usize::try_from(value).map_err(|_| {
            CompressedTensorsError::Invalid(format!(
                "compressed tensor {name} dimension {value} cannot be represented as usize"
            ))
        })?;
    }
    Ok(values)
}

#[derive(Debug)]
pub(crate) enum CompressedTensorsError {
    Artifact(ModelArtifactError),
    Decode(SafetensorsTensorDecodeError),
    Invalid(String),
}

impl fmt::Display for CompressedTensorsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(formatter, "{error}"),
            Self::Decode(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CompressedTensorsError {}

impl From<ModelArtifactError> for CompressedTensorsError {
    fn from(value: ModelArtifactError) -> Self {
        Self::Artifact(value)
    }
}

impl From<SafetensorsTensorDecodeError> for CompressedTensorsError {
    fn from(value: SafetensorsTensorDecodeError) -> Self {
        Self::Decode(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_offset_int4_words_dequantize_by_group() {
        let weight = CompressedTensorsInt4Weight {
            rows: 1,
            columns: 8,
            group_size: 4,
            packed: vec![0xfedc_ba98_u32 as i32],
            scales: vec![0.5, 2.0],
        };

        assert_eq!(
            weight.dequantize_f32(),
            vec![0.0, 0.5, 1.0, 1.5, 8.0, 10.0, 12.0, 14.0]
        );
    }
}
