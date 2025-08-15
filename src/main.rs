#![allow(non_snake_case)]
use std::{collections::HashMap, env};

use burn::{
    backend::{Cuda, NdArray},
    tensor::backend::Backend,
};
use connect4_lib::games::connect4;

use crate::games::{Game, TicTacToe};

mod MCTS;
mod games;
mod tests;

fn game<B: Backend>() {
    let game: Box<dyn Game<B>> = Box::new(TicTacToe::init());
    let args: HashMap<&str, f32> =
        HashMap::from([(("C"), f32::sqrt(2.0)), (("num_searches"), 1000.0)]);

    let mut state = game.get_initial_state();
    let mut player = 1;

    game.print_state(&state);
    loop {
        let mut tree = MCTS::MCTS::new(args.clone(), Box::new(TicTacToe::init()), &state, player);
        let best_action = tree.search();
        state = game.apply_move(&state, player, best_action);
        game.print_state(&state);

        if game.get_value_and_terminated(&state, player).1 {
            break;
        }

        player = -player;
    }
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    env::set_var("RUST_LOG", "debug");
    env_logger::init();
    //let game = connect4();
    //println!("{:?}", game.get_board_layout());

    game::<NdArray>();
}
