use std::{any::Any, ops::Add, string, vec};

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

    pub fn check_win(&self, state: &Tensor<B, 2>, player: i8) -> bool {
        let summed_rows = state.clone().sum_dim(0);
        let summed_collumns = state.clone().sum_dim(1);

        let win_on_any_row: bool = summed_rows
            .clone()
            .equal_elem(player * 3)
            .any()
            .to_data()
            .into_vec()
            .unwrap()[0];

        let win_on_any_col: bool = summed_collumns
            .clone()
            .equal_elem(player * 3)
            .any()
            .to_data()
            .into_vec()
            .unwrap()[0];

        let mut diagonal_index: usize = 0;
        let mut diagonal_index_inversed: usize = 3;
        let mut diagonal_vec: Vec<f32> = Vec::new();
        let mut diagonal_inversed_vec: Vec<f32> = Vec::new();
        for row in state.clone().iter_dim(0) {
            diagonal_index_inversed -= 1;
            let row_vec: Vec<f32> = row.to_data().into_vec().unwrap();
            diagonal_vec.push(row_vec[diagonal_index]);
            diagonal_inversed_vec.push(row_vec[diagonal_index_inversed]);

            diagonal_index += 1;
        }

        let diagonal_win = diagonal_vec.iter().sum::<f32>() == (player * 3) as f32;
        let diagonal_inversed_win =
            diagonal_inversed_vec.iter().sum::<f32>() == (player * 3) as f32;

        return win_on_any_col || win_on_any_row || diagonal_win || diagonal_inversed_win;
    }
}
