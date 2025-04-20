use bincode::impl_borrow_decode;
use burn::tensor::{
    backend::Backend, cast::ToElement, Bool, Float, Int, Shape, Tensor, TensorKind,
};
use rand::{distr::Distribution, rng, Rng};
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{self, Rc, Weak},
    usize,
};

use crate::TicTacToe;

static mut NODE_ID: usize = 0;

#[derive(Debug)]
pub struct Node<'a, B: Backend> {
    game: &'a TicTacToe<B>,
    args: HashMap<&'a str, f32>,
    state: Tensor<B, 2>,
    action_taken: Option<(usize, usize)>,
    expandable_moves: Tensor<B, 2, Bool>,

    pub parent: Weak<RefCell<Self>>,
    children: Vec<Rc<RefCell<Node<'a, B>>>>,

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
        action_taken: Option<(usize, usize)>,
        parent: Weak<RefCell<Self>>,
        player: i8,
    ) -> Node<'a, B> {
        unsafe { NODE_ID += 1 };
        let expandable_moves = game.get_valid_moves_as_mask(&state);

        Node {
            game,
            args,
            state,
            action_taken,
            expandable_moves,

            parent,
            children: Vec::new(),

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

    pub fn select(&'a self) -> &'a RefCell<Node<'a, B>> {
        let mut best_child_index = 0;
        let mut best_ucb: f32 = f32::MIN;

        for (i, child) in self.children.iter().enumerate() {
            let ucb = self.calculate_UCB(&child.borrow());
            if ucb > best_ucb {
                best_child_index = i;
                best_ucb = ucb;
            }
        }

        return &self.children[best_child_index];
    }

    #[allow(non_snake_case)]
    pub fn calculate_UCB(&self, child: &Node<'a, B>) -> f32 {
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

    pub fn expand(self_rc: &Rc<RefCell<Self>>) -> Rc<RefCell<Node<'a, B>>> {
        let mut node = self_rc.borrow_mut();
        let action = node.get_random_action();

        node.expandable_moves = node.expandable_moves.clone().slice_assign(
            [action.0..action.0 + 1, action.1..action.1 + 1],
            Tensor::from([[false]]),
        );

        let mut child_state: Tensor<B, 2> =
            node.game.get_next_state(&node.state, &action, -node.player);
        child_state = node.game.change_perspective(&child_state);

        let child = Rc::new(RefCell::new(Node::new(
            node.game,
            node.args.clone(),
            child_state,
            Some(action),
            Rc::downgrade(self_rc),
            -node.player,
        )));

        node.children.push(Rc::clone(&child));
        child
    }

    fn get_random_action(&self) -> (usize, usize) {
        let valid_moves = self.expandable_moves.clone().argwhere();

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
}
