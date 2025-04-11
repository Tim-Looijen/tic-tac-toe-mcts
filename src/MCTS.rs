use std::{any::Any, collections::HashMap, ops::Add, string, vec};

use bincode::impl_borrow_decode;
use burn::{
    record::Record,
    tensor::{backend::Backend, Bool, Float, Int, Shape, Tensor, TensorKind},
};

use crate::TicTacToe;

static mut NODE_ID: usize = 0;

struct Node<'a, B: Backend> {
    game: &'a TicTacToe<B>,
    args: HashMap<&'a str, f32>,
    state: Tensor<B, 2>,
    parent: Option<&'a Node<'a, B>>,
    action_taken: Option<i8>,
    children: Vec<&'a Node<'a, B>>,
    expandable_moves: Tensor<B, 2, Bool>,

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
        action_taken: Option<i8>,
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

            visit_count: 0,
            value_sum: 0.0,
            id: unsafe { NODE_ID },
        }
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.expandable_moves.clone().all().into_scalar() == false
            && self.children.iter().count() == 0
    }
}
