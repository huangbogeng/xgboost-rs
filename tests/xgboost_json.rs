use std::path::PathBuf;

use xgboost_rs::{DenseMatrix, XgbError, XgbModel};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected) {
        assert!((lhs - rhs).abs() < 1.0e-6);
    }
}

#[test]
fn loads_official_regressor_json_and_predicts() {
    let model = XgbModel::load_json(fixture_path("regressor.json")).unwrap();
    let features = DenseMatrix::from_shape_vec(
        4,
        3,
        vec![
            0.0, 0.0, 0.0, // left-left, left-left
            0.0, 1.0, 0.0, // left-right, left-right
            0.0, -2.0, 1.0, // right-left, right-left
            0.0, 0.0, 1.0, // right-right, right-right
        ],
    )
    .unwrap();

    let predictions = model.predict_dense(&features).unwrap();

    assert_vec_close(
        &predictions,
        &[-0.171_604_96, 0.066_404_61, 0.021_938_96, 0.327_222_02],
    );
}

#[test]
fn loads_official_binary_classifier_json_and_predicts_probabilities() {
    let model = XgbModel::load_json(fixture_path("classifier.json")).unwrap();
    let features = DenseMatrix::from_shape_vec(
        4,
        3,
        vec![
            0.0, 0.0, 0.0, // left-left, left-right
            0.0, 1.0, 0.0, // left-right, left-right
            0.0, -2.0, 1.0, // right-left, left-left
            0.0, 0.0, 1.0, // right-left, right-right
        ],
    )
    .unwrap();

    let predictions = model.predict_dense(&features).unwrap();

    assert_vec_close(
        &predictions,
        &[0.469_360_77, 0.522_249_44, 0.553_298_23, 0.617_173_93],
    );
}

#[test]
fn binary_classifier_reports_probability_base_score() {
    let model = XgbModel::load_json(fixture_path("classifier.json")).unwrap();

    assert!((model.base_score() - 0.52).abs() < 1.0e-6);
}

#[test]
fn xgboost_json_predict_checks_feature_count() {
    let model = XgbModel::load_json(fixture_path("regressor.json")).unwrap();
    let features = DenseMatrix::from_shape_vec(1, 2, vec![0.0, 0.0]).unwrap();

    let error = model.predict_dense(&features).unwrap_err();

    assert!(matches!(
        error,
        XgbError::FeatureCountMismatch {
            expected: 3,
            actual: 2
        }
    ));
}
