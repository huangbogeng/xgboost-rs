use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct JsonModel {
    pub(super) learner: Learner,
}

#[derive(Debug, Deserialize)]
pub(super) struct Learner {
    pub(super) gradient_booster: GradientBooster,
    #[serde(rename = "learner_model_param")]
    pub(super) model_param: LearnerModelParam,
    pub(super) objective: ObjectiveConfig,
}

#[derive(Debug, Deserialize)]
pub(super) struct GradientBooster {
    pub(super) name: String,
    pub(super) model: GbtreeModel,
}

#[derive(Debug, Deserialize)]
pub(super) struct GbtreeModel {
    pub(super) tree_info: Vec<usize>,
    pub(super) trees: Vec<JsonTree>,
}

#[derive(Debug, Deserialize)]
pub(super) struct LearnerModelParam {
    pub(super) base_score: String,
    pub(super) num_feature: String,
    pub(super) num_target: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct ObjectiveConfig {
    pub(super) name: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct JsonTree {
    pub(super) base_weights: Vec<f64>,
    pub(super) default_left: Vec<u8>,
    pub(super) id: usize,
    pub(super) left_children: Vec<i32>,
    pub(super) right_children: Vec<i32>,
    pub(super) split_conditions: Vec<f64>,
    pub(super) split_indices: Vec<u32>,
    pub(super) split_type: Vec<u8>,
    pub(super) tree_param: TreeParam,
}

#[derive(Debug, Deserialize)]
pub(super) struct TreeParam {
    pub(super) num_nodes: String,
    pub(super) size_leaf_vector: String,
}
