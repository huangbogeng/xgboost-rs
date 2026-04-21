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
        return Err(XgbError::InvalidModelFormat(
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
                        return Err(XgbError::InvalidModelFormat("tree contains a cycle"));
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
                            return Err(XgbError::InvalidModelFormat(
                                "left child index out of bounds",
                            ));
                        }
                        if right_child >= num_nodes {
                            return Err(XgbError::InvalidModelFormat(
                                "right child index out of bounds",
                            ));
                        }

                        indegree[left_child] += 1;
                        if indegree[left_child] > 1 {
                            return Err(XgbError::InvalidModelFormat(
                                "tree nodes must have exactly one parent",
                            ));
                        }

                        indegree[right_child] += 1;
                        if indegree[right_child] > 1 {
                            return Err(XgbError::InvalidModelFormat(
                                "tree nodes must have exactly one parent",
                            ));
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
        return Err(XgbError::InvalidModelFormat(
            "tree contains unreachable nodes",
        ));
    }

    Ok(())
}
