use burn::tensor::{
    backend::Backend, cast::ToElement, Bool, Device, Float, Int, Shape, Tensor, TensorKind,
};
use rand::{distr::Distribution, rng, Rng};
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
    usize,
};

use crate::TicTacToe;

#[derive(Debug)]
struct Node<B: Backend> {
    state: Tensor<B, 2>,
    action_taken: Option<(usize, usize)>,
    expandable_moves_as_mask: Tensor<B, 2, Bool>,

    parent_tree_index: Option<usize>,
    children_tree_indices: Vec<usize>,

    player: i8,
    visit_count: u32,
    value_sum: f32,

    pub tree_index: usize,
}

impl<B: Backend> Node<B> {
    pub fn new(
        state: Tensor<B, 2, Float>,
        action_taken: Option<(usize, usize)>,
        expandable_moves_as_mask: Tensor<B, 2, Bool>,

        parent_tree_index: Option<usize>,
        player: i8,

        tree_index: usize,
    ) -> Node<B> {
        Node {
            state,
            action_taken,
            expandable_moves_as_mask,

            parent_tree_index,
            children_tree_indices: Vec::new(),

            player,
            visit_count: 0,
            value_sum: 0.0,

            tree_index,
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.expandable_moves_as_mask
            .clone()
            .any()
            .into_scalar()
            .to_bool()
            == false
            && self.children_tree_indices.iter().count() > 0
    }
}

pub struct AlphaMCTS<'a, B: Backend> {
    game: &'a TicTacToe<B>,
    args: &'a HashMap<&'a str, f32>,
    tree: Vec<Node<B>>,
}

impl<'a, B: Backend> AlphaMCTS<'a, B> {
    pub fn new(game: &'a TicTacToe<B>, args: &'a HashMap<&'a str, f32>) -> AlphaMCTS<'a, B> {
        AlphaMCTS {
            game,
            args,
            tree: vec![],
        }
    }

    pub fn search(&mut self, root_state: &Tensor<B, 2>, player: i8) -> (usize, usize) {
        let root = Node::new(
            root_state.clone(),
            None,
            self.game.get_valid_moves_as_mask(&root_state),
            None,
            player,
            0,
        );
        self.tree.push(root);

        for search in 0..self.args["num_searches"] as usize {
            let mut node_tree_index = self.tree.first().unwrap().tree_index;

            while self.tree[node_tree_index].is_fully_expanded() {
                node_tree_index = self.select(node_tree_index);
            }
            let node = &self.tree[node_tree_index];
            let value_and_terminated = self.game.get_value_and_terminated(&node.state, node.player);

            let mut value = -value_and_terminated.0;
            let is_terminal = value_and_terminated.1;

            if !is_terminal {
                node_tree_index = self.expand(node_tree_index).tree_index;
                value = self.simulate(node_tree_index);
            }

            self.backpropagate(node_tree_index, value);
        }
        let mut action_probs = Tensor::<B, 2>::zeros(
            Shape::new([self.game.row_count, self.game.column_count]),
            &<B as Backend>::Device::default(),
        );
        let root = &self.tree[0];
        for child_tree_index in &root.children_tree_indices {
            let child_tree: usize = child_tree_index.clone();
            let action = self.tree[child_tree].action_taken.unwrap();
            action_probs = action_probs.clone().slice_assign(
                [action.0..action.0 + 1, action.1..action.1 + 1],
                Tensor::from([[self.tree[child_tree].visit_count]]),
            );
        }
        return self.global_argmax(action_probs);
    }

    fn global_argmax(&self, tensor: Tensor<B, 2>) -> (usize, usize) {
        let [rows, cols] = tensor.dims();
        let flattened = tensor.reshape([rows * cols]);
        let index_1d = flattened.argmax(0).into_scalar().to_usize();

        // Convert back to 2D coordinates
        (index_1d / cols, index_1d % cols)
    }

    pub fn select(&self, node_tree_index: usize) -> usize {
        let mut best_child_tree_index = 0;
        let mut best_ucb: f32 = f32::MIN;
        let node = &self.tree[node_tree_index];

        let children: Vec<&Node<B>> = self
            .tree
            .iter()
            .filter(|n| {
                node.children_tree_indices
                    .iter()
                    .any(|&i| i == n.tree_index)
            })
            .collect();

        for child in children.iter() {
            let ucb = self.calculate_UCB(&node, &child);
            if ucb > best_ucb {
                best_child_tree_index = child.tree_index;
                best_ucb = ucb;
            }
        }

        self.tree[best_child_tree_index].tree_index
    }

    pub fn expand(&mut self, node_tree_index: usize) -> &Node<B> {
        let action = self.get_random_action(&self.tree[node_tree_index]);
        self.tree[node_tree_index].expandable_moves_as_mask = self.tree[node_tree_index]
            .expandable_moves_as_mask
            .clone()
            .slice_assign(
                [action.0..action.0 + 1, action.1..action.1 + 1],
                Tensor::from([[false]]),
            );

        let player = &self.tree[node_tree_index].player;

        let mut child_state: Tensor<B, 2> =
            self.game
                .get_next_state(&self.tree[node_tree_index].state, &action, -player);
        child_state = self.game.change_perspective(&child_state);

        let child = Node::new(
            child_state.clone(),
            Some(action),
            self.game.get_valid_moves_as_mask(&child_state),
            Some(node_tree_index),
            -player,
            self.tree.len(),
        );

        self.tree[node_tree_index]
            .children_tree_indices
            .push(child.tree_index);

        self.tree.push(child);

        self.tree.last_mut().unwrap()
    }

    pub fn simulate(&self, node_tree_index: usize) -> f32 {
        let node = &self.tree[node_tree_index];
        let value_terminated = self.game.get_value_and_terminated(&node.state, node.player);

        // Times minus 1 because it is the opponents perspective
        let value: f32 = -value_terminated.0;

        let is_terminal = value_terminated.1;
        if is_terminal {
            return value;
        }

        let mut rollout_state = node.state.clone();
        let mut rollout_player = -node.player;

        loop {
            let action = self.get_random_action(&node);

            rollout_state = self
                .game
                .get_next_state(&rollout_state, &action, rollout_player);

            let (mut value, is_terminal) = self
                .game
                .get_value_and_terminated(&rollout_state, rollout_player);

            if is_terminal {
                if node.player != rollout_player {
                    value = -value;
                }
                return value;
            }

            rollout_player = -rollout_player;
        }
    }

    pub fn backpropagate(&mut self, terminal_tree_index: usize, mut value: f32) {
        self.tree[terminal_tree_index].value_sum += value;
        self.tree[terminal_tree_index].visit_count += 1;

        value = -value;
        let parent_tree_index = self.tree[terminal_tree_index].parent_tree_index;

        if parent_tree_index.is_some() {
            self.backpropagate(parent_tree_index.unwrap(), value);
        }
    }

    fn get_random_action(&self, node: &Node<B>) -> (usize, usize) {
        let valid_moves = node.expandable_moves_as_mask.clone().argwhere();

        let num_indices = valid_moves.dims()[0];
        let chosen_index = rand::random_range(0..num_indices);
        let index_row_col = valid_moves
            .clone()
            .slice([chosen_index..chosen_index + 1])
            .into_data()
            .into_vec::<i32>()
            .unwrap();

        let action: (usize, usize) = (index_row_col[0] as usize, index_row_col[1] as usize);
        return action;
    }

    #[allow(non_snake_case)]
    pub fn calculate_UCB(&self, parent: &Node<B>, child: &Node<B>) -> f32 {
        let w: f32 = child.value_sum;
        let n: f32 = child.visit_count as f32;
        let N: f32 = parent.visit_count as f32;

        // The constant representing the confidence of the model, the higher this value, the less the win_odds matters in the formula
        // Lower means Exploitation, higher means Exploration
        let C: f32 = self.args["C"];

        // avoid devide by zero, this makes the win_odds stay within 0 and 1
        let win_odds: f32 = (w / (n + 1.0)) / 2.0;

        // the q is negative because from the perspective of the parent has the other player, so the state is also inverted
        let q: f32 = -win_odds;

        let UCB: f32 = q + C * f32::sqrt(N.ln_1p() / n);

        return UCB;
    }
}
