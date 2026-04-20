use xgboost_rs::{BoosterTree, DenseMatrix, TreeNode, XgbError, XgbModel};

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected) {
        assert!((lhs - rhs).abs() < f64::EPSILON);
    }
}

#[test]
fn predicts_base_score_without_any_trees() {
    let features = DenseMatrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let model = XgbModel::new(1.5, 2, Vec::new()).unwrap();

    let predictions = model.predict_dense(&features).unwrap();

    assert_vec_close(&predictions, &[1.5, 1.5]);
}

#[test]
fn split_tree_routes_missing_values_and_numeric_values() {
    let features = DenseMatrix::with_missing(3, 1, vec![-1.0, 0.5, 2.0], Some(-1.0)).unwrap();
    let model = XgbModel::new(
        0.25,
        1,
        vec![BoosterTree {
            nodes: vec![
                TreeNode {
                    split_feature: Some(0),
                    split_bin: None,
                    split_value: Some(1.0),
                    left_child: Some(1),
                    right_child: Some(2),
                    leaf_value: None,
                    default_left: true,
                },
                TreeNode::leaf(-0.25),
                TreeNode::leaf(0.75),
            ],
        }],
    )
    .unwrap();

    let predictions = model.predict_dense(&features).unwrap();

    assert_vec_close(&predictions, &[0.0, 0.0, 1.0]);
}

#[test]
fn rejects_feature_count_mismatch() {
    let features = DenseMatrix::from_shape_vec(1, 1, vec![1.0]).unwrap();
    let model = XgbModel::new(0.5, 2, Vec::new()).unwrap();

    let error = model.predict_dense(&features).unwrap_err();

    assert!(matches!(
        error,
        XgbError::FeatureCountMismatch {
            expected: 2,
            actual: 1
        }
    ));
}
