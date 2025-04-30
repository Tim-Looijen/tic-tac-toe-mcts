#![allow(non_snake_case)]
use std::{collections::HashMap, env};

use burn::tensor::backend::Backend;
use tictactoe::TicTacToe;

use self::chess::Chess;

mod MCTS;
mod chess;
mod tests;
mod tictactoe;

fn game<B: Backend>() {
    let game: TicTacToe<B> = TicTacToe::init();
    let args: HashMap<&str, f32> = HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

    let mut state = game.get_initial_state();
    let mut player = -1;

    TicTacToe::print_state(&state);
    loop {
        let mut tree = MCTS::MCTS::new(&game, &args, &state, player);
        let best_action = tree.search();
        state = game.get_next_state(&state, &best_action, player);
        TicTacToe::print_state(&state);
        if game.get_value_and_terminated(&state, player).1 {
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
    Chess::print_state(&Chess::init().game);

    //game::<burn::backend::Cuda>();
}
