use std::any::Any;
use std::fmt::{Debug, Display};

use chess::{Action, Board, ChessMove, Color, Game, GameResult, MoveGen};
use log::{error, warn};

#[derive(Debug, Clone)]
pub struct ChessGame {
    pub game: Game,
}

impl ChessGame {
    pub fn new(game: Game) -> Self {
        Self { game }
    }

    pub fn get_turn(&self) -> Color {
        self.game.side_to_move()
    }

    pub fn apply_move_and_clone(&self, chess_move: ChessMove) -> Self {
        let mut new_game = self.game.clone();
        new_game.make_move(chess_move);
        ChessGame::new(new_game)
    }

    pub fn get_legal_moves(&self) -> MoveGen {
        MoveGen::new_legal(&self.game.current_position())
    }

    pub fn get_value_and_terminated(&self) -> (f32, bool) {
        if self.game.result().is_some() {
            let outcome = self.game.result().unwrap();
            match outcome {
                // Win
                GameResult::WhiteCheckmates
                | GameResult::BlackCheckmates
                | GameResult::WhiteResigns
                | GameResult::BlackResigns => {
                    return (1.0, true);
                }
                // Draw
                _ => {
                    return (0.5, true);
                }
            }
        }

        return (0.0, false);
    }

    pub fn print_state(&self) {
        println!("{}", self.game.current_position());
    }
}
