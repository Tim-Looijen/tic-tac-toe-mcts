use rand::{distr::Distribution, rng, Rng};
use std::{any::Any, collections::HashMap, ops::Add, string, usize, vec};

use bincode::impl_borrow_decode;
use burn::{
    record::Record,
    serde::Serialize,
    tensor::{backend::Backend, cast::ToElement, Bool, Float, Int, Shape, Tensor, TensorKind},
};

use crate::TicTacToe;

static mut NODE_ID: usize = 0;

pub struct Node<'a, B: Backend> {
    game: &'a TicTacToe<B>,
    args: HashMap<&'a str, f32>,
    state: Tensor<B, 2>,
    parent: Option<&'a Node<'a, B>>,
    action_taken: Vec<usize>,
    children: Vec<Node<'a, B>>,
    expandable_moves: Tensor<B, 2, Bool>,

    player: i8,
    visit_count: u32,
    value_sum: f32,
    id: usize,
}

impl<'a, B: Backend> Node<'a, B> {
    pub fn new(
        game: &'a TicTacToe<B>,
        args: HashMap<&'a str, f32>,
        state: Tensor<B, 2, Float>,
        parent: Option<&'a Node<'a, B>>,
        action_taken: Vec<usize>,
        player: i8,
    ) -> Node<'a, B> {
        unsafe { NODE_ID += 1 };
        let expandable_moves = game.get_valid_moves_as_mask(&state);
        Node {
            game,
            args,
            state,
            parent,
            action_taken,
            children: Vec::new(),
            expandable_moves,

            player,
            visit_count: 0,
            value_sum: 0.0,
            id: unsafe { NODE_ID },
        }
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.expandable_moves.clone().all().into_scalar() == false
            && self.children.iter().count() == 0
    }

    pub fn select(&'a self) -> &'a Node<'a, B> {
        let best_child = self;
        return best_child;
    }

    #[allow(non_snake_case)]
    pub fn calculate_UCB(&self, child: &'a Node<'a, B>) -> f32 {
        let w: f32 = child.value_sum;
        let n: f32 = child.visit_count as f32;
        let N: f32 = self.visit_count as f32;

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

    pub fn expand(&'a mut self) -> &'a Node<'a, B> {
        // -> &'a Node<'a, B> {
        let action = self.get_random_action();
        self.expandable_moves = self.expandable_moves.clone().slice_assign(
            [action[0]..action[0] + 1, action[1]..action[1] + 1],
            Tensor::from([[false]]),
        );

        let mut child_state: Tensor<B, 2> =
            self.game.get_next_state(&self.state, &action, -self.player);
        child_state = self.game.change_perspective(&child_state);

        let mut child = Node::new(
            self.game,
            self.args.clone(),
            child_state,
            Some(self),
            action,
            -self.player,
        );

        self.children.push(child);

        return &self.children.last().unwrap();
    }

    fn get_random_action(&self) -> Vec<usize> {
        let valid_moves = self.expandable_moves.clone().argwhere();

        let num_indices = valid_moves.dims()[0];
        let chosen_index = rand::random_range(0..num_indices);
        let index_row_col = valid_moves
            .clone()
            .slice([chosen_index..chosen_index + 1])
            .into_data()
            .into_vec::<i32>()
            .unwrap();

        let action: Vec<usize> = Vec::from([index_row_col[0] as usize, index_row_col[1] as usize]);
        return action;
    }
}
