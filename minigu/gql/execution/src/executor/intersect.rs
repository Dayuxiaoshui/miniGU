use std::collections::HashSet;
use std::sync::Arc;

use minigu_common::value::{ScalarValue, ScalarValueAccessor};

use super::utils::gen_try;
use super::{Executor, IntoExecutor};
use crate::evaluator::datum::DatumRef;
use crate::evaluator::BoxedEvaluator;

#[derive(Debug, PartialEq, Hash, Eq, Clone)]
enum IntersectKey {
    Single(ScalarValue),
    Multi(Vec<ScalarValue>),
}

#[derive(Debug)]
pub struct IntersectBuilder<L, R> {
    left: L,
    right: R,
    left_keys: Vec<BoxedEvaluator>,
    right_keys: Vec<BoxedEvaluator>,
}

impl<L, R> IntersectBuilder<L, R> {
    pub fn new(left: L, right: R, left_keys: Vec<BoxedEvaluator>, right_keys: Vec<BoxedEvaluator>) -> Self {
        assert_eq!(
            left_keys.len(),
            right_keys.len(),
            "left and right keys must have the same length, got {} and {}",
            left_keys.len(),
            right_keys.len()
        );
        assert!(!left_keys.is_empty(), "at least one key column is required");
        Self {
            left,
            right,
            left_keys,
            right_keys,
        }
    }

    pub fn with_single_key(left: L, right: R, left_key: BoxedEvaluator, right_key: BoxedEvaluator) -> Self {
        Self::new(left, right, vec![left_key], vec![right_key])
    }
}

fn make_intersect_key(arrays: &[Arc<dyn arrow::array::Array>], row: usize) -> IntersectKey {
    if arrays.len() == 1 {
        IntersectKey::Single(arrays[0].as_ref().index(row))
    } else {
        let mut keys = Vec::with_capacity(arrays.len());
        for arr in arrays {
            keys.push(arr.as_ref().index(row));
        }
        IntersectKey::Multi(keys)
    }
}

fn key_has_null(key: &IntersectKey) -> bool {
    match key {
        IntersectKey::Single(v) => matches!(v, ScalarValue::Null),
        IntersectKey::Multi(values) => values.iter().any(|v| matches!(v, ScalarValue::Null)),
    }
}

impl<L, R> IntoExecutor for IntersectBuilder<L, R>
where
    L: Executor,
    R: Executor,
{
    type IntoExecutor = impl Executor;

    fn into_executor(self) -> Self::IntoExecutor {
        gen move {
            let IntersectBuilder {
                left,
                right,
                left_keys,
                right_keys,
            } = self;

            let mut left_set: HashSet<IntersectKey> = HashSet::new();
            let mut left_has_data = false;

            for chunk in left.into_iter() {
                let chunk = gen_try!(chunk);
                if chunk.is_empty() {
                    continue;
                }

                let key_arrays: Vec<_> = gen_try!(
                    left_keys
                        .iter()
                        .map(|e| e.evaluate(&chunk).map(DatumRef::into_array))
                        .collect::<Result<Vec<_>, _>>()
                );

                for row in 0..chunk.len() {
                    let key = make_intersect_key(&key_arrays, row);
                    if !key_has_null(&key) {
                        left_set.insert(key);
                        left_has_data = true;
                    }
                }
            }

            if !left_has_data {
                return;
            }

            for chunk in right.into_iter() {
                let chunk = gen_try!(chunk);
                if chunk.is_empty() {
                    continue;
                }

                let key_arrays: Vec<_> = gen_try!(
                    right_keys
                        .iter()
                        .map(|e| e.evaluate(&chunk).map(DatumRef::into_array))
                        .collect::<Result<Vec<_>, _>>()
                );

                let mut intersected_rows = Vec::new();

                for row in 0..chunk.len() {
                    let key = make_intersect_key(&key_arrays, row);
                    if !key_has_null(&key) && left_set.contains(&key) {
                        intersected_rows.push(row);
                        left_set.remove(&key);
                    }
                }

                if !intersected_rows.is_empty() {
                    let indices = arrow::array::UInt32Array::from(
                        intersected_rows.into_iter().map(|r| r as u32).collect::<Vec<_>>()
                    );
                    let result_chunk = chunk.take(&indices);
                    yield Ok(result_chunk);
                }
            }
        }
        .into_executor()
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, AsArray};
    use minigu_common::data_chunk;
    use minigu_common::data_chunk::DataChunk;

    use super::*;
    use crate::evaluator::column_ref::ColumnRef;

    #[test]
    fn test_intersect_basic() {
        let left_chunk = data_chunk!((UInt64, [1, 2, 3, 4]));
        let right_chunk = data_chunk!((UInt64, [3, 4, 5, 6]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let intersect_executor = IntersectBuilder::with_single_key(
            left_executor,
            right_executor,
            Box::new(ColumnRef::new(0)),
            Box::new(ColumnRef::new(0)),
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];

        let result_array = result.columns()[0].as_primitive::<arrow::datatypes::UInt64Type>();
        let mut values: Vec<u64> = (0..result_array.len())
            .map(|i| result_array.value(i))
            .collect();
        values.sort_unstable();

        assert_eq!(values, vec![3, 4]);
    }

    #[test]
    fn test_intersect_strings() {
        let left_chunk = data_chunk!((Utf8, ["Alice", "Bob", "Charlie"]));
        let right_chunk = data_chunk!((Utf8, ["Bob", "David", "Alice"]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let intersect_executor = IntersectBuilder::with_single_key(
            left_executor,
            right_executor,
            Box::new(ColumnRef::new(0)),
            Box::new(ColumnRef::new(0)),
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];

        let result_array = result.columns()[0].as_string::<i32>();
        let mut values: Vec<&str> = (0..result_array.len())
            .map(|i| result_array.value(i))
            .collect();
        values.sort_unstable();

        assert_eq!(values, vec!["Alice", "Bob"]);
    }

    #[test]
    fn test_intersect_multi_column() {
        let left_chunk = data_chunk!(
            (Int32, [1, 2, 3]),
            (Utf8, ["a", "b", "c"])
        );
        let right_chunk = data_chunk!(
            (Int32, [2, 3, 4]),
            (Utf8, ["b", "c", "d"])
        );

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let intersect_executor = IntersectBuilder::new(
            left_executor,
            right_executor,
            vec![Box::new(ColumnRef::new(0)), Box::new(ColumnRef::new(1))],
            vec![Box::new(ColumnRef::new(0)), Box::new(ColumnRef::new(1))],
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];

        let col0 = result.columns()[0].as_primitive::<arrow::datatypes::Int32Type>();
        let col1 = result.columns()[1].as_string::<i32>();

        let values: Vec<(i32, &str)> = (0..result.len())
            .map(|i| (col0.value(i), col1.value(i)))
            .collect();

        assert_eq!(values, vec![(2, "b"), (3, "c")]);
    }

    #[test]
    fn test_intersect_no_match() {
        let left_chunk = data_chunk!((UInt64, [1, 2]));
        let right_chunk = data_chunk!((UInt64, [3, 4]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let intersect_executor = IntersectBuilder::with_single_key(
            left_executor,
            right_executor,
            Box::new(ColumnRef::new(0)),
            Box::new(ColumnRef::new(0)),
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_intersect_with_nulls() {
        let left_chunk = data_chunk!((Int32, [Some(1), Some(2), None, Some(3)]));
        let right_chunk = data_chunk!((Int32, [Some(2), None, Some(3), Some(4)]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let intersect_executor = IntersectBuilder::with_single_key(
            left_executor,
            right_executor,
            Box::new(ColumnRef::new(0)),
            Box::new(ColumnRef::new(0)),
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];

        let result_array = result.columns()[0].as_primitive::<arrow::datatypes::Int32Type>();
        let values: Vec<i32> = (0..result_array.len())
            .filter_map(|i| {
                if !result_array.is_null(i) {
                    Some(result_array.value(i))
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(values, vec![2, 3]);
    }

    #[test]
    fn test_intersect_with_duplicates() {
        let left_chunk = data_chunk!((UInt64, [1, 2, 3]));
        let right_chunk = data_chunk!((UInt64, [2, 2, 3, 3]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let intersect_executor = IntersectBuilder::with_single_key(
            left_executor,
            right_executor,
            Box::new(ColumnRef::new(0)),
            Box::new(ColumnRef::new(0)),
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];

        let result_array = result.columns()[0].as_primitive::<arrow::datatypes::UInt64Type>();
        let mut values: Vec<u64> = (0..result_array.len())
            .map(|i| result_array.value(i))
            .collect();
        values.sort_unstable();

        assert_eq!(values, vec![2, 3]);
    }

    #[test]
    fn test_intersect_multiple_chunks() {
        let left_chunks = vec![
            data_chunk!((UInt64, [1, 2])),
            data_chunk!((UInt64, [3, 4])),
        ];
        let right_chunks = vec![
            data_chunk!((UInt64, [2, 3])),
            data_chunk!((UInt64, [4, 5])),
        ];

        let left_executor = left_chunks.into_iter().map(Ok).into_executor();
        let right_executor = right_chunks.into_iter().map(Ok).into_executor();

        let intersect_executor = IntersectBuilder::with_single_key(
            left_executor,
            right_executor,
            Box::new(ColumnRef::new(0)),
            Box::new(ColumnRef::new(0)),
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let mut all_values = Vec::new();
        for chunk in results {
            let array = chunk.columns()[0].as_primitive::<arrow::datatypes::UInt64Type>();
            for i in 0..array.len() {
                all_values.push(array.value(i));
            }
        }
        all_values.sort_unstable();

        assert_eq!(all_values, vec![2, 3, 4]);
    }

    #[test]
    fn test_intersect_empty_left() {
        let left_chunk = data_chunk!((UInt64, [None, None, None]));
        let right_chunk = data_chunk!((UInt64, [1, 2, 3]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let intersect_executor = IntersectBuilder::with_single_key(
            left_executor,
            right_executor,
            Box::new(ColumnRef::new(0)),
            Box::new(ColumnRef::new(0)),
        )
        .into_executor();

        let results: Vec<DataChunk> = intersect_executor
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    #[should_panic(expected = "left and right keys must have the same length")]
    fn test_intersect_mismatched_key_length() {
        let left_chunk = data_chunk!((UInt64, [1, 2, 3]));
        let right_chunk = data_chunk!((UInt64, [2, 3, 4]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let _ = IntersectBuilder::new(
            left_executor,
            right_executor,
            vec![Box::new(ColumnRef::new(0)), Box::new(ColumnRef::new(1))],
            vec![Box::new(ColumnRef::new(0))],
        );
    }

    #[test]
    #[should_panic(expected = "at least one key column is required")]
    fn test_intersect_empty_keys() {
        let left_chunk = data_chunk!((UInt64, [1, 2, 3]));
        let right_chunk = data_chunk!((UInt64, [2, 3, 4]));

        let left_executor = [Ok(left_chunk)].into_executor();
        let right_executor = [Ok(right_chunk)].into_executor();

        let _ = IntersectBuilder::new(
            left_executor,
            right_executor,
            vec![],
            vec![],
        );
    }
}
