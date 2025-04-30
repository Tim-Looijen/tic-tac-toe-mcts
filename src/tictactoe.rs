use std::fmt::Debug;

use burn::module::ModuleDisplay;
use burn::tensor::Device;
use burn::tensor::{backend::Backend, cast::ToElement, Float, Tensor};

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

    pub fn get_legal_moves(&self, state: &Tensor<B, 2, Float>) -> Vec<(usize, usize)> {
        let legal_moves_as_mask = state.clone().equal_elem(0);
        if !legal_moves_as_mask.clone().any().into_scalar().to_bool() {
            return vec![];
        }

        legal_moves_as_mask
            .argwhere()
            .into_data()
            .to_vec::<i32>()
            .unwrap()
            .chunks_exact(2)
            .map(|pair| (pair[0] as usize, pair[1] as usize))
            .collect()
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

    pub fn get_value_and_terminated(&self, state: &Tensor<B, 2, Float>, player: i8) -> (f32, bool) {
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

    pub fn print_state(state: &Tensor<B, 2>) {
        let data = state.clone().into_data();
        let slice: &[f32] = data.as_slice().unwrap();
        let [rows, cols] = [data.shape[0], data.shape[1]];

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
        println!("<======>");
    }

    pub fn change_perspective(&self, state: &Tensor<B, 2>) -> Tensor<B, 2> {
        return state.clone().mul(Tensor::from([[-1]]));
    }
}
