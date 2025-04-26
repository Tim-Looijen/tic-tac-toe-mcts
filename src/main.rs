use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    rc::{Rc, Weak},
};

use burn::tensor::{backend::Backend, Tensor};
use tictactoe::TicTacToe;
use MCTS::AlphaMCTS;
mod MCTS;
mod tictactoe;

fn game<B: Backend>() {
    let game: TicTacToe<B> = TicTacToe::init();
    let args: HashMap<&str, f32> = HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

    let mut state = game.get_initial_state();
    let mut player = 1;
    loop {
        let mut tree = AlphaMCTS::new(&game, &args);
        let action = tree.search(&state.clone(), player);
        println!("{:?}", vec![action]);
        state = game.get_next_state(&state, &action, player);
        println!("{:}", state);

        if game.check_win(&state, player) {
            println!("{:} Won", player);
            return;
        }
        player = -player;
    }
}

fn main() {
    if true {
        env::set_var("RUST_BACKTRACE", "1");
    }

    game::<burn::backend::Cuda>();
}
