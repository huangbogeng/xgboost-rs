use std::fs;
use std::path::PathBuf;

use xgboost_rs::{DenseMatrix, XgbModel};

fn fixture_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(path)
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected) {
        assert!((lhs - rhs).abs() < 1.0e-6, "left={lhs}, right={rhs}");
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
    let header = &rows[0];
    let n_cols = header.len();

    let mut data = Vec::new();
    for row in rows.iter().skip(1) {
        assert_eq!(row.len(), n_cols);
        for value in row {
            data.push(value.parse::<f64>().unwrap());
        }
    }

    DenseMatrix::from_shape_vec(rows.len() - 1, n_cols, data).unwrap()
}

fn read_softprob_expected(path: &str) -> (Vec<f64>, Vec<usize>) {
    let rows = read_csv_rows(fixture_path(path));
    let header = &rows[0];
    assert_eq!(
        header,
        &[
            "prob_class_0",
            "prob_class_1",
            "prob_class_2",
            "predicted_label"
        ]
    );

    let mut probabilities = Vec::new();
    let mut labels = Vec::new();
    for row in rows.iter().skip(1) {
        assert_eq!(row.len(), 4);
        probabilities.push(row[0].parse::<f64>().unwrap());
        probabilities.push(row[1].parse::<f64>().unwrap());
        probabilities.push(row[2].parse::<f64>().unwrap());
        labels.push(row[3].parse::<usize>().unwrap());
    }

    (probabilities, labels)
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

fn argmax_index(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

#[test]
fn loads_official_multiclass_softprob_json_and_predicts_probabilities() {
    let model =
        XgbModel::load_json(fixture_path("softprob/iris_3class_softprob_v1.model.json")).unwrap();
    let features = read_dense_features("softprob/cache_X_test.csv");
    let (expected_probabilities, expected_labels) =
        read_softprob_expected("softprob/cache_y_pred.csv");

    let predictions = model.predict_dense(&features).unwrap();

    assert_eq!(model.base_margins().len(), 3);
    assert_eq!(predictions.len(), expected_probabilities.len());
    assert_vec_close(&predictions, &expected_probabilities);

    let predicted_labels = predictions
        .chunks_exact(3)
        .map(argmax_index)
        .collect::<Vec<_>>();
    assert_eq!(predicted_labels, expected_labels);

    for row_probs in predictions.chunks_exact(3) {
        let probability_sum = row_probs.iter().sum::<f64>();
        assert!((probability_sum - 1.0).abs() < 1.0e-6);
    }
}

#[test]
fn loads_official_multiclass_softmax_json_and_predicts_labels() {
    let model =
        XgbModel::load_json(fixture_path("softmax/iris_3class_softmax_v1.model.json")).unwrap();
    let features = read_dense_features("softmax/cache_X_test.csv");
    let expected_labels = read_single_float_column("softmax/cache_y_pred.csv", "predicted_label");
    let expected_true = read_single_float_column("softmax/cache_y_true.csv", "true_label");

    let predictions = model.predict_dense(&features).unwrap();

    assert_eq!(model.base_margins().len(), 3);
    assert_eq!(predictions.len(), expected_labels.len());
    assert_vec_close(&predictions, &expected_labels);
    assert_vec_close(&predictions, &expected_true);
}
