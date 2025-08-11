#![allow(non_snake_case)]
use std::{collections::HashMap, env};

use burn::tensor::backend::Backend;
use connect4_lib::{ games::connect4};

use crate::{
    games::{Game, Player, TicTacToe},
    MCTS::MCTS,
};

mod MCTS;
//mod tests;
mod games;

fn game<B: Backend>() {
    let game: TicTacToe<B> = TicTacToe::init();
    let args: HashMap<&str, f32> = HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

    let mut state = game.get_initial_state();
    let mut player = Player::;

    game.print_state(&state);
    loop {
        let mut tree = MCTS::MCTS::new(&args, state.clone());
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
    let game = connect4();
    println!("{:?}", game.get_board_layout());

    //   game::<burn::backend::Cuda>();
}
