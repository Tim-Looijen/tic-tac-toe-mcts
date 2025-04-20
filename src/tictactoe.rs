use std::{any::Any, ops::Add, string, vec};

use burn::{
    record::Record,
    tensor::{backend::Backend, Bool, Float, Int, Shape, Tensor, TensorKind},
};

#[derive(Debug)]
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
        action: &(usize, usize),
        player: i8,
    ) -> Tensor<B, 2> {
        let next_state = previous_state.clone();

        let row = action.0;
        let column = action.1;
        let player_tensor = Tensor::from_floats([[player]], &next_state.device());

        next_state.slice_assign([row..row + 1, column..column + 1], player_tensor)
    }

    pub fn get_valid_moves_as_mask(&self, state: &Tensor<B, 2, Float>) -> Tensor<B, 2, Bool> {
        let valid_moves = state.clone();
        let mask = valid_moves.equal_elem(0);

        return mask;
    }

    pub fn check_win(&self, state: &Tensor<B, 2>, player: i8) -> bool {
        let summed_rows = state.clone().sum_dim(0);
        let summed_collumns = state.clone().sum_dim(1);

        let win_on_any_row: bool = summed_rows
            .clone()
            .equal_elem(player * 3)
            .any()
            .into_scalar();

        let win_on_any_col: bool = summed_collumns
            .clone()
            .equal_elem(player * 3)
            .any()
            .into_scalar();

        let mut diagonal_index: usize = 0;
        let mut diagonal_index_inversed: usize = 3;
        let mut diagonal_summed: i8 = 0;
        let mut diagonal_inversed_summed: i8 = 0;
        for row in state.clone().iter_dim(0) {
            diagonal_index_inversed -= 1;
            let row_vec: Vec<f32> = row.to_data().into_vec().unwrap();
            diagonal_summed += row_vec[diagonal_index] as i8;
            diagonal_inversed_summed += row_vec[diagonal_index_inversed] as i8;

            diagonal_index += 1;
        }

        let diagonal_win = diagonal_summed == (player * 3);
        let diagonal_inversed_win = diagonal_inversed_summed == (player * 3);

        return win_on_any_col || win_on_any_row || diagonal_win || diagonal_inversed_win;
    }

    pub fn change_perspective(&self, state: &Tensor<B, 2>) -> Tensor<B, 2> {
        return state.clone().mul(Tensor::from([[-1]]));
    }
}
