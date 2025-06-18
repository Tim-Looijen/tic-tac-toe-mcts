#![allow(non_snake_case)]
use std::{collections::HashMap, env};

use self::chess_game::ChessGame;
use burn::tensor::backend::Backend;
use chess::Game;
use tictactoe::TicTacToe;

mod MCTS;
mod chess_game;
mod tests;
mod tictactoe;

fn game<B: Backend>() {
    let game: ChessGame = ChessGame::new(Game::new());
    let args: HashMap<&str, f32> = HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

    let mut state = game.clone();
    let mut player = -1;

    state.print_state();
    loop {
        let mut tree = MCTS::AlphaMCTS::new(&args, state.clone());
        let best_action = tree.search();
        state = state.apply_move_and_clone(best_action);
        state.print_state();

        if game.get_value_and_terminated().1 {
            break;
        }

        player = -player;
    }
}

fn main() {
    if 1 == 1 {
        env::set_var("RUST_BACKTRACE", "1");
    }
    env::set_var("RUST_LOG", "info");
    env_logger::init();

    game::<burn::backend::Cuda>();
}
