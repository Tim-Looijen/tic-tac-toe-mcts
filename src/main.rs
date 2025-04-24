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
    let test = game.get_initial_state();
    let test2 = game.get_next_state(&test, &(0, 0), -1);
    let test3 = game.get_next_state(&test2, &(0, 1), -1);
    let test4 = game.get_next_state(&test3, &(0, 2), -1);
    println!("{:}", test);
    println!("{:}", test2);
    println!("{:}", test3);
    println!("{:}", test4);
    println!("{:}", game.check_win(&test4, -1));

    let args: HashMap<&str, f32> =
        HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 10000.0)]);

    let tree = AlphaMCTS::new(&game, args);
}

fn main() {
    game::<burn::backend::CudaJit>();
}
