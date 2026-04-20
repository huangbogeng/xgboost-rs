//! Loading support for official upstream `XGBoost` model files.

mod loader;
mod schema;

pub(crate) use loader::load_json_model;

#[cfg(test)]
mod tests {
    use super::loader::convert_model;
    use super::schema::JsonModel;
    use crate::error::{Result, XGBError};

    fn load_model_from_json(json: &str) -> Result<crate::model::XGBModel> {
        let model: JsonModel = serde_json::from_str(json)?;
        convert_model(model)
    }

    fn wrap_single_tree(tree_json: &str) -> String {
        format!(
            r#"{{
  "learner": {{
    "gradient_booster": {{
      "name": "gbtree",
      "model": {{
        "tree_info": [0],
        "trees": [{tree_json}]
      }}
    }},
    "learner_model_param": {{
      "base_score": "0.5",
      "num_feature": "1",
      "num_target": "1"
    }},
    "objective": {{
      "name": "reg:squarederror"
    }}
  }}
}}"#
        )
    }

    #[test]
    fn rejects_empty_tree_structure() {
        let json = wrap_single_tree(
            r#"{
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
    "size_leaf_vector": "1"
  }
}"#,
        );

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XGBError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_tree_cycles() {
        let json = wrap_single_tree(
            r#"{
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
    "size_leaf_vector": "1"
  }
}"#,
        );

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XGBError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_unreachable_tree_nodes() {
        let json = wrap_single_tree(
            r#"{
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
    "size_leaf_vector": "1"
  }
}"#,
        );

        let error = load_model_from_json(&json).unwrap_err();

        assert!(matches!(error, XGBError::InvalidModelFormat(_)));
    }
}
