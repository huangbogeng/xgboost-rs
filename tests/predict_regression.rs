use xgboost_rs::{DMatrix, DenseMatrix, XGBRegressor};

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected) {
        assert!((lhs - rhs).abs() < f64::EPSILON);
    }
}

#[test]
fn predict_uses_base_score_when_no_trees_exist() {
    let features = DenseMatrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let train = DMatrix::from_dense(features.clone(), vec![1.5, 1.5]).unwrap();

    let model = XGBRegressor::builder().base_score(1.5).build().unwrap();
    let fitted = model.fit(&train).unwrap();
    let predictions = fitted.predict_dense(&features).unwrap();

    assert_vec_close(&predictions, &[1.5, 1.5]);
}
