use std::fmt::{Debug, Display};

use burn::module::ModuleDisplay;
use burn::tensor::Device;
use burn::tensor::{backend::Backend, cast::ToElement, Float, Tensor};
use rchess::{self, ChessBoard, ChessGame, MoveGen, Square};

#[derive(Debug)]
pub struct Chess {
    pub game: ChessGame,
}

impl Chess {
    pub fn init() -> Self {
        Self {
            game: ChessGame::new(),
        }
    }

    pub fn get_initial_state(&self) {}

    pub fn get_next_state(&self, move_piece: Square, to: Square) {}

    pub fn get_legal_moves(&self) -> MoveGen {
        return MoveGen::legal(self.game.board());
    }

    pub fn check_win(&self) -> bool {
        return self.game.result().is_some();
    }

    pub fn get_value_and_terminated(&self) -> (f32, bool) {
        // win
        if self.check_win() {
            return (1.0, true);
        }

        // draw
        if self.get_legal_moves().len() == 0 {
            return (0.5, true);
        }

        return (0.0, false);
    }

    pub fn print_state(game: &ChessGame) {
        println!("{}", game.board());
    }
}
