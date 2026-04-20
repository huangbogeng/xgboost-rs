use std::fs;
use std::path::PathBuf;

use xgboost_rs::{DenseMatrix, XgbError, XgbModel};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected) {
        assert!(
            (lhs - rhs).abs() <= tolerance,
            "left={lhs}, right={rhs}, tolerance={tolerance}"
        );
    }
}

fn read_csv_rows(path: PathBuf) -> Vec<Vec<String>> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split(',').map(str::to_owned).collect())
        .collect()
}

fn read_dense_features(path: &str) -> DenseMatrix {
    let rows = read_csv_rows(fixture_path(path));
    let n_cols = rows[0].len();

    let mut data = Vec::new();
    for row in rows.iter().skip(1) {
        assert_eq!(row.len(), n_cols);
        for value in row {
            data.push(value.parse::<f64>().unwrap());
        }
    }

    DenseMatrix::from_shape_vec(rows.len() - 1, n_cols, data).unwrap()
}

fn read_single_float_column(path: &str, expected_header: &str) -> Vec<f64> {
    let rows = read_csv_rows(fixture_path(path));
    assert_eq!(rows[0].as_slice(), [expected_header]);

    rows.iter()
        .skip(1)
        .map(|row| {
            assert_eq!(row.len(), 1);
            row[0].parse::<f64>().unwrap()
        })
        .collect()
}

fn read_binary_predictions(path: &str) -> (Vec<f64>, Vec<f64>) {
    let rows = read_csv_rows(fixture_path(path));
    assert_eq!(rows[0].as_slice(), ["prob_class_1", "predicted_label"]);

    let mut probabilities = Vec::new();
    let mut labels = Vec::new();
    for row in rows.iter().skip(1) {
        assert_eq!(row.len(), 2);
        probabilities.push(row[0].parse::<f64>().unwrap());
        labels.push(row[1].parse::<f64>().unwrap());
    }

    (probabilities, labels)
}

fn labels_from_binary_probability(probabilities: &[f64]) -> Vec<f64> {
    probabilities
        .iter()
        .map(|value| if *value >= 0.5 { 1.0 } else { 0.0 })
        .collect()
}

fn mean_absolute_error(actual: &[f64], expected: &[f64]) -> f64 {
    let sample_count = u32::try_from(actual.len()).unwrap();
    actual
        .iter()
        .zip(expected)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f64>()
        / f64::from(sample_count)
}

fn max_absolute_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn accuracy_at_least(
    predicted_labels: &[f64],
    true_labels: &[f64],
    numerator: usize,
    denominator: usize,
) -> bool {
    let correct = predicted_labels
        .iter()
        .zip(true_labels)
        .filter(|(predicted, truth)| (**predicted - **truth).abs() <= f64::EPSILON)
        .count();

    correct * denominator >= predicted_labels.len() * numerator
}

#[test]
fn loads_official_regressor_json_and_predicts() {
    let model = XgbModel::load_json(fixture_path("regression/model.json")).unwrap();
    let features = read_dense_features("regression/cache_X_test.csv");
    let expected = read_single_float_column("regression/cache_y_pred.csv", "predicted_value");

    let predictions = model.predict_dense(&features).unwrap();

    assert_eq!(predictions.len(), expected.len());

    let mae = mean_absolute_error(&predictions, &expected);
    let max_error = max_absolute_error(&predictions, &expected);

    assert!(mae <= 3.0, "mae={mae}");
    assert!(max_error <= 25.0, "max_error={max_error}");
}

#[test]
fn loads_official_binary_classifier_json_and_predicts_probabilities() {
    let model = XgbModel::load_json(fixture_path("binary/model.json")).unwrap();
    let features = read_dense_features("binary/cache_X_test.csv");
    let (expected_probabilities, expected_labels) =
        read_binary_predictions("binary/cache_y_pred.csv");

    let predictions = model.predict_dense(&features).unwrap();

    assert_vec_close(&predictions, &expected_probabilities, 1.0e-6);

    let predicted_labels = labels_from_binary_probability(&predictions);
    assert_vec_close(&predicted_labels, &expected_labels, 0.0);
}

#[test]
fn binary_classifier_reports_probability_base_score() {
    let model = XgbModel::load_json(fixture_path("binary/model.json")).unwrap();

    assert!((model.base_score() - 0.628_571_45).abs() < 1.0e-6);
}

#[test]
fn xgboost_json_predict_checks_feature_count() {
    let model = XgbModel::load_json(fixture_path("regression/model.json")).unwrap();
    let features = DenseMatrix::from_shape_vec(1, 9, vec![0.0; 9]).unwrap();

    let error = model.predict_dense(&features).unwrap_err();

    assert!(matches!(
        error,
        XgbError::FeatureCountMismatch {
            expected: 10,
            actual: 9
        }
    ));
}

#[test]
fn binary_fixture_probability_threshold_matches_exported_labels() {
    let (expected_probabilities, expected_labels) =
        read_binary_predictions("binary/cache_y_pred.csv");
    let threshold_labels = labels_from_binary_probability(&expected_probabilities);

    assert_vec_close(&threshold_labels, &expected_labels, 0.0);
}

#[test]
fn binary_fixture_predictions_reach_expected_accuracy_against_true_labels() {
    let model = XgbModel::load_json(fixture_path("binary/model.json")).unwrap();
    let features = read_dense_features("binary/cache_X_test.csv");
    let true_labels = read_single_float_column("binary/cache_y_true.csv", "true_label");

    let probabilities = model.predict_dense(&features).unwrap();
    let predicted_labels = labels_from_binary_probability(&probabilities);

    assert_eq!(predicted_labels.len(), true_labels.len());
    assert!(accuracy_at_least(&predicted_labels, &true_labels, 95, 100));
}
