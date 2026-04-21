use crate::error::{Result, XgbError};

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Unvisited,
    Visiting,
    Visited,
}

#[derive(Clone, Copy)]
enum TraversalStep {
    Enter(usize),
    Exit(usize),
}

pub(crate) fn validate_tree_topology<F>(num_nodes: usize, mut children_of: F) -> Result<()>
where
    F: FnMut(usize) -> Result<Option<(usize, usize)>>,
{
    if num_nodes == 0 {
        return Err(XgbError::invalid_model_format(
            "trees must contain at least one node",
        ));
    }

    let mut visit_state = vec![VisitState::Unvisited; num_nodes];
    let mut indegree = vec![0usize; num_nodes];
    let mut stack = vec![TraversalStep::Enter(0)];

    while let Some(step) = stack.pop() {
        match step {
            TraversalStep::Enter(node_idx) => {
                match visit_state[node_idx] {
                    VisitState::Unvisited => {}
                    VisitState::Visiting => {
                        return Err(XgbError::invalid_model_format(format!(
                            "tree contains a cycle at node index {node_idx}",
                        )));
                    }
                    VisitState::Visited => continue,
                }

                visit_state[node_idx] = VisitState::Visiting;
                match children_of(node_idx)? {
                    None => {
                        visit_state[node_idx] = VisitState::Visited;
                    }
                    Some((left_child, right_child)) => {
                        if left_child >= num_nodes {
                            return Err(XgbError::invalid_model_format(format!(
                                "left child index out of bounds at node {node_idx}: left_child={left_child}, node_count={num_nodes}",
                            )));
                        }
                        if right_child >= num_nodes {
                            return Err(XgbError::invalid_model_format(format!(
                                "right child index out of bounds at node {node_idx}: right_child={right_child}, node_count={num_nodes}",
                            )));
                        }

                        indegree[left_child] += 1;
                        if indegree[left_child] > 1 {
                            return Err(XgbError::invalid_model_format(format!(
                                "tree nodes must have exactly one parent: node {left_child} has multiple parents",
                            )));
                        }

                        indegree[right_child] += 1;
                        if indegree[right_child] > 1 {
                            return Err(XgbError::invalid_model_format(format!(
                                "tree nodes must have exactly one parent: node {right_child} has multiple parents",
                            )));
                        }

                        stack.push(TraversalStep::Exit(node_idx));
                        stack.push(TraversalStep::Enter(right_child));
                        stack.push(TraversalStep::Enter(left_child));
                    }
                }
            }
            TraversalStep::Exit(node_idx) => {
                visit_state[node_idx] = VisitState::Visited;
            }
        }
    }

    if visit_state.contains(&VisitState::Unvisited) {
        let first_unreachable = visit_state
            .iter()
            .position(|state| *state == VisitState::Unvisited)
            .unwrap_or(0);
        return Err(XgbError::invalid_model_format(format!(
            "tree contains unreachable nodes starting at node index {first_unreachable}",
        )));
    }

    Ok(())
}
