use std::{any::Any, string};

use burn::{
    record::Record,
    tensor::{backend::Backend, Bool, Int, Shape, Tensor, TensorKind},
};

pub struct TicTacToe<B: Backend> {
    pub row_count: usize,
    pub column_count: usize,
    pub play_space: usize,
    pub action_size: usize,
    pub device: <B as Backend>::Device,
    pub title: &'static str,
}

impl<B: Backend> TicTacToe<B> {
    pub fn init() -> TicTacToe<B> {
        TicTacToe {
            row_count: 3,
            column_count: 3,
            play_space: 9,
            action_size: 9,
            device: Default::default(),
            title: "TicTacToe",
        }
    }

    pub fn get_initial_state(&self) -> Tensor<B, 2> {
        let initial_state: Tensor<B, 2> =
            Tensor::zeros([self.row_count, self.row_count], &self.device);
        return initial_state;
    }

    pub fn get_next_state(
        &self,
        previous_state: &Tensor<B, 2>,
        action: i32,
        player: i8,
    ) -> Tensor<B, 2> {
        let next_state = previous_state.clone();

        let row: usize = action as usize / self.row_count;
        let column: usize = action as usize % self.column_count;
        let player_tensor = Tensor::from_floats([[player]], &next_state.device());

        next_state.slice_assign([row..row + 1, column..column + 1], player_tensor)
    }

    pub fn get_valid_moves_as_mask(state: &Tensor<B, 2>) -> Tensor<B, 2, Bool> {
        let valid_moves = state.clone();
        let mask = valid_moves.equal_elem(0);

        return mask;
    }

    pub fn check_win(&self, state: &Tensor<B, 2>, action: i32) -> bool {
        if action == -1 {
            return false;
        }
        let state = state.clone();

        let row: usize = action as usize / self.row_count;
        let column: usize = action as usize % self.column_count;

        let test = state.sum_dim(0);

        println!("{:}", test);

        return true;
    }
}
