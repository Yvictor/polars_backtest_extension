//! FFI conversion between polars-arrow and arrow-rs
//!
//! This module provides zero-copy conversion from polars-arrow arrays to arrow-rs arrays
//! using the Arrow C Data Interface.

use arrow::array::{Array as ArrowArray, ArrayRef, Float64Array, Int32Array, StringViewArray};
use arrow::ffi as ar_ffi;
use polars_arrow::array::Array as PolarsArray;
use polars_arrow::datatypes::Field;
use polars_arrow::ffi as pa_ffi;

/// Convert a polars-arrow array to an arrow-rs ArrayRef using FFI
///
/// This conversion is zero-copy for the underlying data buffers.
/// The Arc reference counting ensures memory safety.
///
/// # Safety
/// This function uses unsafe FFI operations but is safe to call because:
/// - Both polars-arrow and arrow-rs implement the Arrow C Data Interface
/// - Memory ownership is properly transferred via the C interface
pub fn polars_to_arrow_rs(array: Box<dyn PolarsArray>) -> Result<ArrayRef, arrow::error::ArrowError> {
    unsafe {
        // Create a Field from the array's data type (for schema export)
        let field = Field::new("".into(), array.dtype().clone(), true);

        // Export from polars-arrow to C structs
        let pa_array = pa_ffi::export_array_to_c(array);
        let pa_schema = pa_ffi::export_field_to_c(&field);

        // The C structs from polars-arrow and arrow-rs have the same layout
        // (both implement Arrow C Data Interface), so we can transmute
        let ar_array: ar_ffi::FFI_ArrowArray = std::mem::transmute(pa_array);
        let ar_schema: ar_ffi::FFI_ArrowSchema = std::mem::transmute(pa_schema);

        // Import to arrow-rs
        let array_data = ar_ffi::from_ffi(ar_array, &ar_schema)?;
        Ok(arrow::array::make_array(array_data))
    }
}

/// Convert polars-arrow PrimitiveArray<i32> to arrow-rs Int32Array
pub fn polars_i32_to_arrow(
    array: &polars_arrow::array::PrimitiveArray<i32>,
) -> Result<Int32Array, arrow::error::ArrowError> {
    let boxed = array.to_boxed();
    let array_ref = polars_to_arrow_rs(boxed)?;

    // Downcast to Int32Array
    array_ref
        .as_any()
        .downcast_ref::<Int32Array>()
        .cloned()
        .ok_or_else(|| {
            arrow::error::ArrowError::CastError("Failed to downcast to Int32Array".to_string())
        })
}

/// Convert polars-arrow PrimitiveArray<f64> to arrow-rs Float64Array
pub fn polars_f64_to_arrow(
    array: &polars_arrow::array::PrimitiveArray<f64>,
) -> Result<Float64Array, arrow::error::ArrowError> {
    let boxed = array.to_boxed();
    let array_ref = polars_to_arrow_rs(boxed)?;

    array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| {
            arrow::error::ArrowError::CastError("Failed to downcast to Float64Array".to_string())
        })
}

/// Convert polars-arrow Utf8ViewArray to arrow-rs StringViewArray (zero-copy)
pub fn polars_utf8view_to_arrow(
    array: &polars_arrow::array::Utf8ViewArray,
) -> Result<StringViewArray, arrow::error::ArrowError> {
    let boxed = array.to_boxed();
    let array_ref = polars_to_arrow_rs(boxed)?;

    // Utf8ViewArray converts to StringViewArray via FFI (zero-copy)
    array_ref
        .as_any()
        .downcast_ref::<StringViewArray>()
        .cloned()
        .ok_or_else(|| {
            arrow::error::ArrowError::CastError(
                format!("Failed to downcast to StringViewArray. Actual type: {:?}", array_ref.data_type())
            )
        })
}

/// Check that FFI struct sizes match (for safe transmute)
/// Returns (polars_array_size, polars_schema_size, arrow_array_size, arrow_schema_size)
#[allow(dead_code)]
pub fn check_ffi_struct_sizes() -> (usize, usize, usize, usize) {
    (
        std::mem::size_of::<pa_ffi::ArrowArray>(),
        std::mem::size_of::<pa_ffi::ArrowSchema>(),
        std::mem::size_of::<ar_ffi::FFI_ArrowArray>(),
        std::mem::size_of::<ar_ffi::FFI_ArrowSchema>(),
    )
}

/// Verify FFI compatibility at runtime
#[allow(dead_code)]
pub fn verify_ffi_compatibility() -> Result<(), String> {
    let (pa_array, pa_schema, ar_array, ar_schema) = check_ffi_struct_sizes();

    if pa_array != ar_array {
        return Err(format!(
            "ArrowArray size mismatch: polars-arrow={}, arrow-rs={}",
            pa_array, ar_array
        ));
    }
    if pa_schema != ar_schema {
        return Err(format!(
            "ArrowSchema size mismatch: polars-arrow={}, arrow-rs={}",
            pa_schema, ar_schema
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars_arrow::array::PrimitiveArray;

    #[test]
    fn test_ffi_struct_sizes() {
        let (pa_array, pa_schema, ar_array, ar_schema) = check_ffi_struct_sizes();
        println!("polars-arrow ArrowArray: {} bytes", pa_array);
        println!("polars-arrow ArrowSchema: {} bytes", pa_schema);
        println!("arrow-rs FFI_ArrowArray: {} bytes", ar_array);
        println!("arrow-rs FFI_ArrowSchema: {} bytes", ar_schema);

        // These should match for safe transmute
        assert_eq!(pa_array, ar_array, "ArrowArray size mismatch");
        assert_eq!(pa_schema, ar_schema, "ArrowSchema size mismatch");
    }

    #[test]
    fn test_i32_conversion() {
        let pa_array = PrimitiveArray::<i32>::from_vec(vec![1, 2, 3, 4, 5]);
        let ar_array = polars_i32_to_arrow(&pa_array).unwrap();

        assert_eq!(ar_array.len(), 5);
        assert_eq!(ar_array.value(0), 1);
        assert_eq!(ar_array.value(4), 5);
    }

    #[test]
    fn test_f64_conversion() {
        let pa_array = PrimitiveArray::<f64>::from_vec(vec![1.0, 2.5, 3.14]);
        let ar_array = polars_f64_to_arrow(&pa_array).unwrap();

        assert_eq!(ar_array.len(), 3);
        assert!((ar_array.value(0) - 1.0).abs() < 1e-10);
        assert!((ar_array.value(2) - 3.14).abs() < 1e-10);
    }
}
