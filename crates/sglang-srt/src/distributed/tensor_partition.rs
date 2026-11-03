use std::fs;
use std::io::{Read, Seek, SeekFrom};

use crate::model_artifacts::{
    ModelArtifactError, SafetensorsTensorData, SafetensorsTensorMetadata, SafetensorsTensorSpan,
};
use crate::parallel::TensorPartition;

pub fn read_tensor_axis_partition(
    span: &SafetensorsTensorSpan,
    axis: usize,
    partition: TensorPartition,
) -> Result<SafetensorsTensorData, ModelArtifactError> {
    let axis_extent = span.metadata.shape.get(axis).copied().ok_or_else(|| {
        invalid_partition(
            span,
            format!(
                "tensor rank {} has no axis {axis}",
                span.metadata.shape.len()
            ),
        )
    })?;
    let range = partition
        .range(axis_extent)
        .map_err(|error| invalid_partition(span, error.to_string()))?;
    let dtype_byte_width = span.metadata.dtype_byte_width().ok_or_else(|| {
        invalid_partition(
            span,
            format!(
                "tensor dtype {} has no fixed byte width",
                span.metadata.dtype
            ),
        )
    })?;
    let element_count = span
        .metadata
        .element_count()
        .ok_or_else(|| invalid_partition(span, "tensor element count overflowed".to_string()))?;
    let expected_byte_len = element_count
        .checked_mul(dtype_byte_width)
        .ok_or_else(|| invalid_partition(span, "tensor byte length overflowed".to_string()))?;
    if expected_byte_len != span.byte_len {
        return Err(invalid_partition(
            span,
            format!(
                "tensor metadata describes {expected_byte_len} bytes but span contains {}",
                span.byte_len
            ),
        ));
    }

    let outer_count = checked_shape_product(span, &span.metadata.shape[..axis])?;
    let inner_count = checked_shape_product(span, &span.metadata.shape[axis + 1..])?;
    let source_outer_stride = axis_extent
        .checked_mul(inner_count)
        .ok_or_else(|| invalid_partition(span, "source tensor stride overflowed".to_string()))?;
    let selected_axis_len = range.end - range.start;
    let selected_outer_elements = selected_axis_len
        .checked_mul(inner_count)
        .ok_or_else(|| invalid_partition(span, "partition stride overflowed".to_string()))?;
    let selected_outer_bytes = selected_outer_elements
        .checked_mul(dtype_byte_width)
        .ok_or_else(|| invalid_partition(span, "partition byte stride overflowed".to_string()))?;
    let selected_byte_len = outer_count
        .checked_mul(selected_outer_bytes)
        .ok_or_else(|| invalid_partition(span, "partition byte length overflowed".to_string()))?;
    let mut bytes = vec![0_u8; selected_byte_len];
    let mut file =
        fs::File::open(&span.path).map_err(|error| ModelArtifactError::ReadWeightShard {
            path: span.path.clone(),
            message: error.to_string(),
        })?;

    for outer_index in 0..outer_count {
        let source_element_offset = outer_index
            .checked_mul(source_outer_stride)
            .and_then(|offset| {
                range
                    .start
                    .checked_mul(inner_count)
                    .and_then(|start| offset.checked_add(start))
            })
            .ok_or_else(|| {
                invalid_partition(span, "partition source offset overflowed".to_string())
            })?;
        let source_byte_offset = source_element_offset
            .checked_mul(dtype_byte_width)
            .ok_or_else(|| {
                invalid_partition(span, "partition source byte offset overflowed".to_string())
            })?;
        let absolute_byte_offset = span
            .absolute_byte_offset
            .checked_add(u64::try_from(source_byte_offset).map_err(|_| {
                invalid_partition(span, "partition source offset exceeds u64".to_string())
            })?)
            .ok_or_else(|| {
                invalid_partition(span, "partition absolute offset overflowed".to_string())
            })?;
        file.seek(SeekFrom::Start(absolute_byte_offset))
            .map_err(|error| ModelArtifactError::ReadWeightShard {
                path: span.path.clone(),
                message: error.to_string(),
            })?;
        let target_start = outer_index
            .checked_mul(selected_outer_bytes)
            .ok_or_else(|| {
                invalid_partition(span, "partition target offset overflowed".to_string())
            })?;
        let target_end = target_start
            .checked_add(selected_outer_bytes)
            .ok_or_else(|| {
                invalid_partition(span, "partition target end overflowed".to_string())
            })?;
        file.read_exact(&mut bytes[target_start..target_end])
            .map_err(|error| ModelArtifactError::ReadWeightShard {
                path: span.path.clone(),
                message: error.to_string(),
            })?;
    }

    let mut shape = span.metadata.shape.clone();
    shape[axis] = selected_axis_len;
    Ok(SafetensorsTensorData {
        metadata: SafetensorsTensorMetadata {
            dtype: span.metadata.dtype.clone(),
            shape,
            data_offsets: [0, bytes.len()],
        },
        bytes,
    })
}

fn checked_shape_product(
    span: &SafetensorsTensorSpan,
    dimensions: &[usize],
) -> Result<usize, ModelArtifactError> {
    dimensions.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| invalid_partition(span, "tensor shape product overflowed".to_string()))
    })
}

fn invalid_partition(span: &SafetensorsTensorSpan, message: String) -> ModelArtifactError {
    ModelArtifactError::InvalidSafetensorsData {
        path: span.path.clone(),
        message: format!("invalid tensor partition: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn reads_row_and_column_partitions_without_materializing_the_full_tensor() {
        let path = fixture_path("axis-slices");
        let prefix = [0xAA_u8; 17];
        let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let tensor_bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut file_bytes = prefix.to_vec();
        file_bytes.extend_from_slice(&tensor_bytes);
        fs::write(&path, file_bytes).expect("write tensor fixture");
        let span = SafetensorsTensorSpan {
            path: path.clone(),
            metadata: SafetensorsTensorMetadata {
                dtype: "F32".to_string(),
                shape: vec![4, 6],
                data_offsets: [0, tensor_bytes.len()],
            },
            absolute_byte_offset: prefix.len() as u64,
            byte_len: tensor_bytes.len(),
        };

        let rows = read_tensor_axis_partition(
            &span,
            0,
            TensorPartition::new(2, 1).expect("row partition"),
        )
        .expect("read row partition");
        assert_eq!(rows.metadata.shape, vec![2, 6]);
        assert_eq!(
            rows.decode_f32_values().expect("decode rows"),
            values[12..].to_vec()
        );

        let columns = read_tensor_axis_partition(
            &span,
            1,
            TensorPartition::new(3, 1).expect("column partition"),
        )
        .expect("read column partition");
        assert_eq!(columns.metadata.shape, vec![4, 2]);
        assert_eq!(
            columns.decode_f32_values().expect("decode columns"),
            vec![2.0, 3.0, 8.0, 9.0, 14.0, 15.0, 20.0, 21.0]
        );

        fs::remove_file(path).expect("remove tensor fixture");
    }

    #[test]
    fn rejects_invalid_axis_and_uneven_partition_before_file_io() {
        let span = SafetensorsTensorSpan {
            path: fixture_path("missing"),
            metadata: SafetensorsTensorMetadata {
                dtype: "BF16".to_string(),
                shape: vec![3, 4],
                data_offsets: [0, 24],
            },
            absolute_byte_offset: 0,
            byte_len: 24,
        };

        let invalid_axis =
            read_tensor_axis_partition(&span, 2, TensorPartition::new(1, 0).expect("partition"))
                .expect_err("invalid axis");
        assert!(invalid_axis.to_string().contains("has no axis 2"));

        let uneven =
            read_tensor_axis_partition(&span, 0, TensorPartition::new(2, 0).expect("partition"))
                .expect_err("uneven partition");
        assert!(uneven.to_string().contains("must be divisible"));
    }

    fn fixture_path(name: &str) -> std::path::PathBuf {
        let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "sglang-rs-tensor-partition-{name}-{}-{id}.bin",
            std::process::id()
        ))
    }
}
