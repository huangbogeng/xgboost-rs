use xgboost_rs::{DMatrix, DenseMatrix, XGBRegressor};

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected) {
        assert!((lhs - rhs).abs() < f64::EPSILON);
    }
}

#[test]
fn fit_initializes_base_score_from_label_mean() {
    let features = DenseMatrix::from_shape_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let train = DMatrix::from_dense(features, vec![2.0, 4.0, 6.0]).unwrap();

    let model = XGBRegressor::builder().build().unwrap();
    let fitted = model.fit(&train).unwrap();

    assert!((fitted.base_score() - 4.0).abs() < f64::EPSILON);
    assert_eq!(fitted.n_features(), 1);
}

#[test]
fn single_tree_learns_a_step_function() {
    let features = DenseMatrix::from_shape_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let train = DMatrix::from_dense(features.clone(), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

    let model = XGBRegressor::builder()
        .n_estimators(1)
        .max_depth(1)
        .learning_rate(1.0)
        .max_bin(4)
        .lambda(0.0)
        .gamma(0.0)
        .min_child_weight(0.0)
        .build()
        .unwrap();

    let fitted = model.fit(&train).unwrap();
    let predictions = fitted.predict_dense(train.features()).unwrap();

    assert_vec_close(&predictions, &[0.0, 0.0, 1.0, 1.0]);
}
