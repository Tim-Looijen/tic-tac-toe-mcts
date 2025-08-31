use std::{error::Error, fmt::Debug, num::IntErrorKind};

use burn::tensor::{backend::Backend, cast::ToElement, Float, Tensor};
use log::debug;

#[derive(Debug)]
pub struct TicTacToe<B: Backend> {
    pub row_count: usize,
    pub column_count: usize,
    pub play_space: usize,
    pub action_size: usize,
    device: <B as Backend>::Device,
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

    pub fn check_win(&self, state: &Tensor<B, 2>, player: i8) -> bool {
        let summed_rows = state.clone().sum_dim(0);
        let summed_collumns = state.clone().sum_dim(1);

        let win_on_any_row: bool = summed_rows
            .equal_elem(player * 3)
            .any()
            .into_scalar()
            .to_bool();

        let win_on_any_col: bool = summed_collumns
            .equal_elem(player * 3)
            .any()
            .into_scalar()
            .to_bool();

        let mut diagonal_summed: i8 = 0;
        let mut diagonal_inversed_summed: i8 = 0;

        for (i, row) in state.clone().iter_dim(0).enumerate() {
            let row_vec: Vec<f32> = row.to_data().into_vec().unwrap();
            diagonal_summed += row_vec[i] as i8;
            diagonal_inversed_summed += row_vec[2 - i] as i8;
        }

        let diagonal_win = diagonal_summed == (player * 3);
        let diagonal_inversed_win = diagonal_inversed_summed == (player * 3);

        return win_on_any_col || win_on_any_row || diagonal_win || diagonal_inversed_win;
    }

    pub fn create_state(&self, player_coordinates: Vec<(usize, usize, i8)>) -> Tensor<B, 2> {
        let mut state = self.get_initial_state();
        for coordinate_player in player_coordinates {
            let row = coordinate_player.0;
            let column = coordinate_player.1;
            state = self.apply_move(&state, coordinate_player.2, (row, column));
        }

        state
    }

    pub fn invert_perspective(&self, state: &Tensor<B, 2>) -> Tensor<B, 2> {
        return state.clone().mul(Tensor::from([[-1]]));
    }

    pub fn get_initial_state(&self) -> Tensor<B, 2> {
        let initial_state: Tensor<B, 2> =
            Tensor::zeros([self.row_count, self.column_count], &self.device);
        return initial_state;
    }

    pub fn apply_move(
        &self,
        state: &Tensor<B, 2>,
        player: i8,
        action: (usize, usize),
    ) -> Tensor<B, 2> {
        let next_state = state.clone();

        let row = action.0;
        let column = action.1;

        let player_tensor = Tensor::from_floats([[player]], &self.device);

        next_state.slice_assign([row..row + 1, column..column + 1], player_tensor)
    }

    pub fn get_value_and_terminated(&self, state: &Tensor<B, 2>, player: i8) -> (f32, bool) {
        // win
        if self.check_win(state, player) {
            return (1.0, true);
        }

        // draw
        if self.get_legal_moves(state).len() == 0 {
            return (0.5, true);
        }

        // lose
        return (0.0, false);
    }

    pub fn get_legal_moves(&self, state: &Tensor<B, 2, Float>) -> Vec<(usize, usize)> {
        let legal_moves_as_mask = state.clone().equal_elem(0);
        if !legal_moves_as_mask.clone().any().into_scalar().to_bool() {
            return vec![];
        }

        legal_moves_as_mask
            .argwhere()
            .into_data()
            .to_vec::<i64>()
            .unwrap()
            .chunks_exact(2)
            .map(|pair| (pair[0] as usize, pair[1] as usize))
            .collect()
    }

    pub fn print_state(&self, state: &Tensor<B, 2>) {
        let data = state.clone().into_data();
        let slice: &[f32] = data.as_slice().unwrap();
        let [rows, cols] = [data.shape[0], data.shape[1]];

        println!();
        for i in 0..rows {
            for j in 0..cols {
                let cell = slice[i * cols + j];
                if cell > 0.0 {
                    print!("X");
                } else if cell < 0.0 {
                    print!("O");
                } else {
                    print!("-");
                }
                if j + 1 < cols {
                    print!("  ");
                }
            }
            println!();
        }
        println!();
    }
}
