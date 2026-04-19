use std::time::{SystemTime, UNIX_EPOCH};

use xgboost_rs::{DMatrix, DenseMatrix, XGBRegressor};

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected) {
        assert!((lhs - rhs).abs() < f64::EPSILON);
    }
}

#[test]
fn model_can_round_trip_through_json() {
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
        .unwrap()
        .fit(&train)
        .unwrap();

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("xgboost-rs-model-{unique}.json"));

    model.save_json(&path).unwrap();
    let loaded = XGBRegressor::load_json(&path).unwrap();

    let original_predictions = model.predict_dense(&features).unwrap();
    let loaded_predictions = loaded.predict_dense(&features).unwrap();
    assert_vec_close(&original_predictions, &loaded_predictions);

    std::fs::remove_file(path).unwrap();
}
