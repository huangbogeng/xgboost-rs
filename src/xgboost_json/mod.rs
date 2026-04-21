//! Loading support for official upstream `XGBoost` model files.

mod loader;
mod schema;

pub(crate) use loader::load_model_json;

#[cfg(test)]
mod tests {
    use super::loader::build_model_from_json;
    use super::schema::JsonModel;
    use crate::dataset::DenseMatrix;
    use crate::error::{Result, XgbError};
    use serde_json::{Value, json};

    fn load_model_from_json(json: &str) -> Result<crate::model::XgbModel> {
        let model: JsonModel = serde_json::from_str(json)?;
        build_model_from_json(model)
    }

    fn wrap_single_tree(
        tree_json: &Value,
        objective: &str,
        base_score: &str,
        num_class: usize,
        num_feature: usize,
        num_target: usize,
        tree_info: &[usize],
    ) -> String {
        json!({
            "learner": {
                "gradient_booster": {
                    "name": "gbtree",
                    "model": {
                        "tree_info": tree_info,
                        "trees": [tree_json],
                    },
                },
                "learner_model_param": {
                    "base_score": base_score,
                    "num_class": num_class.to_string(),
                    "num_feature": num_feature.to_string(),
                    "num_target": num_target.to_string(),
                },
                "objective": {
                    "name": objective,
                },
            },
        })
        .to_string()
    }

    fn wrap_regression_single_tree(tree_json: &Value) -> String {
        wrap_single_tree(tree_json, "reg:squarederror", "0.5", 0, 1, 1, &[0])
    }

    fn leaf_tree(leaf_value: f64) -> Value {
        json!({
            "base_weights": [leaf_value],
            "default_left": [0],
            "id": 0,
            "left_children": [-1],
            "right_children": [-1],
            "split_conditions": [leaf_value],
            "split_indices": [0],
            "split_type": [0],
            "tree_param": {
                "num_nodes": "1",
                "size_leaf_vector": "1",
            },
        })
    }

    fn split_tree_with_default_left(default_left: u8) -> Value {
        json!({
            "base_weights": [0.0, -2.0, 2.0],
            "default_left": [default_left, 0, 0],
            "id": 0,
            "left_children": [1, -1, -1],
            "right_children": [2, -1, -1],
            "split_conditions": [1.0, -2.0, 2.0],
            "split_indices": [0, 0, 0],
            "split_type": [0, 0, 0],
            "tree_param": {
                "num_nodes": "3",
                "size_leaf_vector": "1",
            },
        })
    }

    fn assert_vec_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected) {
            assert!((lhs - rhs).abs() < 1.0e-12, "left={lhs}, right={rhs}");
        }
    }

    fn deep_left_chain_tree(depth: usize) -> Value {
        let num_nodes = depth
            .checked_mul(2)
            .and_then(|node_count| node_count.checked_add(1))
            .expect("depth should fit in usize");

        let mut base_weights = vec![0.0_f64; num_nodes];
        let default_left = vec![0_u8; num_nodes];
        let mut left_children = vec![-1_i32; num_nodes];
        let mut right_children = vec![-1_i32; num_nodes];
        let split_conditions = vec![0.0_f64; num_nodes];
        let split_indices = vec![0_u32; num_nodes];
        let split_type = vec![0_u8; num_nodes];

        for node_idx in 0..depth {
            let left_child = if node_idx + 1 < depth {
                node_idx + 1
            } else {
                depth
            };
            let right_child = depth + 1 + node_idx;

            left_children[node_idx] =
                i32::try_from(left_child).expect("child index should fit in i32");
            right_children[node_idx] =
                i32::try_from(right_child).expect("child index should fit in i32");
        }

        base_weights[depth] = 7.0;
        for value in base_weights.iter_mut().skip(depth + 1) {
            *value = -1.0;
        }

        json!({
            "base_weights": base_weights,
            "default_left": default_left,
            "id": 0,
            "left_children": left_children,
            "right_children": right_children,
            "split_conditions": split_conditions,
            "split_indices": split_indices,
            "split_type": split_type,
            "tree_param": {
                "num_nodes": num_nodes.to_string(),
                "size_leaf_vector": "1",
            },
        })
    }

    #[test]
    fn rejects_empty_tree_structure() {
        let json = wrap_regression_single_tree(&json!({
            "base_weights": [],
            "default_left": [],
            "id": 0,
            "left_children": [],
            "right_children": [],
            "split_conditions": [],
            "split_indices": [],
            "split_type": [],
            "tree_param": {
                "num_nodes": "0",
                "size_leaf_vector": "1",
            },
        }));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_unsupported_gradient_booster() {
        let mut model: Value = serde_json::from_str(&wrap_single_tree(
            &leaf_tree(0.0),
            "reg:squarederror",
            "0.5",
            0,
            1,
            1,
            &[0],
        ))
        .unwrap();
        model["learner"]["gradient_booster"]["name"] = json!("gblinear");
        let json = model.to_string();

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(
            error,
            XgbError::UnsupportedModel {
                context: "gradient booster",
                ..
            }
        ));
    }

    #[test]
    fn rejects_unsupported_objective() {
        let json = wrap_single_tree(&leaf_tree(0.0), "rank:pairwise", "0.5", 0, 1, 1, &[0]);

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(
            error,
            XgbError::UnsupportedModel {
                context: "objective",
                ..
            }
        ));
    }

    #[test]
    fn rejects_multi_target_models() {
        let json = wrap_single_tree(&leaf_tree(0.0), "reg:squarederror", "0.5", 0, 1, 2, &[0]);

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(
            error,
            XgbError::UnsupportedModel {
                context: "num_target",
                ..
            }
        ));
    }

    #[test]
    fn rejects_unsupported_split_type() {
        let json = wrap_regression_single_tree(&json!({
            "base_weights": [0.0, -1.0, 1.0],
            "default_left": [0, 0, 0],
            "id": 0,
            "left_children": [1, -1, -1],
            "right_children": [2, -1, -1],
            "split_conditions": [1.0, -1.0, 1.0],
            "split_indices": [0, 0, 0],
            "split_type": [1, 0, 0],
            "tree_param": {
                "num_nodes": "3",
                "size_leaf_vector": "1",
            },
        }));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(
            error,
            XgbError::UnsupportedModel {
                context: "split_type",
                ..
            }
        ));
    }

    #[test]
    fn rejects_leaf_vector_size_larger_than_one() {
        let json = wrap_regression_single_tree(&json!({
            "base_weights": [0.0],
            "default_left": [0],
            "id": 0,
            "left_children": [-1],
            "right_children": [-1],
            "split_conditions": [0.0],
            "split_indices": [0],
            "split_type": [0],
            "tree_param": {
                "num_nodes": "1",
                "size_leaf_vector": "2",
            },
        }));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(
            error,
            XgbError::UnsupportedModel {
                context: "tree_param.size_leaf_vector",
                ..
            }
        ));
    }

    #[test]
    fn rejects_tree_id_position_mismatch() {
        let json = wrap_regression_single_tree(&json!({
            "base_weights": [0.0],
            "default_left": [0],
            "id": 1,
            "left_children": [-1],
            "right_children": [-1],
            "split_conditions": [0.0],
            "split_indices": [0],
            "split_type": [0],
            "tree_param": {
                "num_nodes": "1",
                "size_leaf_vector": "1",
            },
        }));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_tree_cycles() {
        let json = wrap_regression_single_tree(&json!({
            "base_weights": [0.0, 1.0],
            "default_left": [0, 0],
            "id": 0,
            "left_children": [0, -1],
            "right_children": [1, -1],
            "split_conditions": [0.5, 1.0],
            "split_indices": [0, 0],
            "split_type": [0, 0],
            "tree_param": {
                "num_nodes": "2",
                "size_leaf_vector": "1",
            },
        }));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_unreachable_tree_nodes() {
        let json = wrap_regression_single_tree(&json!({
            "base_weights": [0.0, 1.0, 2.0, 3.0],
            "default_left": [0, 0, 0, 0],
            "id": 0,
            "left_children": [1, -1, -1, -1],
            "right_children": [2, -1, -1, -1],
            "split_conditions": [0.5, 1.0, 2.0, 3.0],
            "split_indices": [0, 0, 0, 0],
            "split_type": [0, 0, 0, 0],
            "tree_param": {
                "num_nodes": "4",
                "size_leaf_vector": "1",
            },
        }));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_invalid_default_left_encoding() {
        let json = wrap_regression_single_tree(&split_tree_with_default_left(2));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_split_nodes_with_missing_child() {
        let json = wrap_regression_single_tree(&json!({
            "base_weights": [0.0, 1.0],
            "default_left": [0, 0],
            "id": 0,
            "left_children": [1, -1],
            "right_children": [-1, -1],
            "split_conditions": [0.5, 1.0],
            "split_indices": [0, 0],
            "split_type": [0, 0],
            "tree_param": {
                "num_nodes": "2",
                "size_leaf_vector": "1",
            },
        }));

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_tree_info_length_mismatch() {
        let json = wrap_single_tree(&leaf_tree(0.0), "reg:squarederror", "0.0", 0, 1, 1, &[]);

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_out_of_range_split_feature_index() {
        let json = wrap_single_tree(
            &json!({
                "base_weights": [0.0, -1.0, 1.0],
                "default_left": [0, 0, 0],
                "id": 0,
                "left_children": [1, -1, -1],
                "right_children": [2, -1, -1],
                "split_conditions": [1.0, -1.0, 1.0],
                "split_indices": [1, 0, 0],
                "split_type": [0, 0, 0],
                "tree_param": {
                    "num_nodes": "3",
                    "size_leaf_vector": "1",
                },
            }),
            "reg:squarederror",
            "0.0",
            0,
            1,
            1,
            &[0],
        );

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_binary_logistic_base_score_boundaries() {
        for boundary in ["0.0", "1.0"] {
            let json =
                wrap_single_tree(&leaf_tree(0.0), "binary:logistic", boundary, 0, 1, 1, &[0]);

            let error = load_model_from_json(&json).unwrap_err();

            assert!(matches!(error, XgbError::InvalidModelFormat(_)));
        }
    }

    #[test]
    fn rejects_multiclass_base_score_length_mismatch() {
        let json = wrap_single_tree(
            &leaf_tree(0.0),
            "multi:softprob",
            "[0.1, 0.2]",
            3,
            1,
            1,
            &[0],
        );

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(
            error,
            XgbError::InvalidShape {
                context: "base_score",
                expected: 3,
                actual: 2,
            }
        ));
    }

    #[test]
    fn routes_nan_missing_values_when_default_left_is_true() {
        let json = wrap_regression_single_tree(&split_tree_with_default_left(1));
        let model = load_model_from_json(&json).unwrap();
        let features = DenseMatrix::from_shape_vec(3, 1, vec![f64::NAN, 0.5, 2.0]).unwrap();

        let predictions = model.predict_dense(&features).unwrap();

        assert_vec_close(&predictions, &[-1.5, -1.5, 2.5]);
    }

    #[test]
    fn routes_nan_missing_values_when_default_left_is_false() {
        let json = wrap_regression_single_tree(&split_tree_with_default_left(0));
        let model = load_model_from_json(&json).unwrap();
        let features = DenseMatrix::from_shape_vec(3, 1, vec![f64::NAN, 0.5, 2.0]).unwrap();

        let predictions = model.predict_dense(&features).unwrap();

        assert_vec_close(&predictions, &[2.5, -1.5, 2.5]);
    }

    #[test]
    fn loads_deep_unbalanced_tree_and_predicts() {
        let json = wrap_single_tree(
            &deep_left_chain_tree(2_048),
            "reg:squarederror",
            "0.0",
            0,
            1,
            1,
            &[0],
        );
        let model = load_model_from_json(&json).unwrap();
        let features = DenseMatrix::from_shape_vec(1, 1, vec![-1.0]).unwrap();

        let predictions = model.predict_dense(&features).unwrap();

        assert_vec_close(&predictions, &[7.0]);
    }
}
