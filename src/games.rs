use std::fmt::Debug;

use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};

#[derive(Debug)]
pub struct TicTacToe {
    pub row_count: usize,
    pub column_count: usize,
}

impl TicTacToe {
    pub fn init() -> TicTacToe {
        TicTacToe {
            row_count: 3,
            column_count: 3,
        }
    }

    pub fn check_win(&self, state: &Array2<i8>, player: i8) -> bool {
        let state = state.clone();
        let summed_rows = state.sum_axis(Axis(0));
        let summed_collumns = state.sum_axis(Axis(1));

        let win_on_any_row: bool = summed_rows.iter().any(|&x| x == player * 3);
        let win_on_any_col: bool = summed_collumns.iter().any(|&x| x == player * 3);

        let mut diagonal_summed: i8 = 0;
        let mut diagonal_inversed_summed: i8 = 0;

        for (i, row) in state.axis_iter(Axis(0)).enumerate() {
            let row_vec = row.to_vec();
            diagonal_summed += row_vec[i];
            diagonal_inversed_summed += row_vec[2 - i];
        }

        let diagonal_win = diagonal_summed == (player * 3);
        let diagonal_inversed_win = diagonal_inversed_summed == (player * 3);

        win_on_any_col || win_on_any_row || diagonal_win || diagonal_inversed_win
    }

    pub fn create_state(&self, player_coordinates: Vec<(usize, usize, i8)>) -> Array2<i8> {
        let mut state = self.get_initial_state();
        for coordinate_player in player_coordinates {
            let row = coordinate_player.0;
            let column = coordinate_player.1;
            state = self.apply_move(&state, coordinate_player.2, (row, column));
        }

        state
    }

    pub fn get_initial_state(&self) -> Array2<i8> {
        Array2::<i8>::zeros([self.row_count, self.column_count])
    }

    pub fn apply_move(&self, state: &Array2<i8>, player: i8, action: (usize, usize)) -> Array2<i8> {
        let mut next_state = state.clone();

        let row = action.0;
        let column = action.1;

        next_state[[row, column]] = player;
        next_state
    }

    pub fn get_value_and_terminated(&self, state: &Array2<i8>, player: i8) -> (f32, bool) {
        // win
        if self.check_win(state, player) {
            return (1.0, true);
        }

        // draw
        if self.get_legal_moves(state).is_empty() {
            return (0.5, true);
        }

        // lose
        (0.0, false)
    }

    pub fn get_legal_moves(&self, state: &Array2<i8>) -> Vec<(usize, usize)> {
        let legal_moves_as_mask: Array2<bool> = state.clone().map(|&x| x == 0);
        if !legal_moves_as_mask.iter().any(|&x| x) {
            return vec![];
        }
        legal_moves_as_mask
            .indexed_iter()
            .filter_map(|(idx, &val)| if val { Some((idx.0, idx.1)) } else { None })
            .collect()
    }

    pub fn print_state(&self, state: &Array2<i8>) -> Result<()> {
        let data = state.clone();
        let slice: &[i8] = data
            .as_slice()
            .ok_or(anyhow!("Unable to slice this array: {:?}", state))?;
        let [rows, cols] = [data.shape()[0], data.shape()[1]];

        println!();
        for i in 0..rows {
            for j in 0..cols {
                let cell = slice[i * cols + j];
                if cell > 0 {
                    print!("X");
                } else if cell < 0 {
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
        Ok(())
    }
}
