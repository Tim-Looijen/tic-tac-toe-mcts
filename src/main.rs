use std::env;

use burn::tensor::{backend::Backend, Tensor};
use tictactoe::TicTacToe;
mod tictactoe;

fn game<B: Backend>() {
    let game: TicTacToe<B> = TicTacToe::init();
    let test = game.get_initial_state();
    let test2 = game.get_next_state(&test, 2, 1);
    let test3 = game.get_next_state(&test2, 4, 1);
    let test4 = game.get_next_state(&test3, 7, 1);
    println!("{:}", test);
    println!("{:}", test2);
    println!("{:}", test3);
    println!("{:}", test4);
    println!("{:}", game.check_win(&test4, 1));
}

fn main() {
    game::<burn::backend::CudaJit>();
}
