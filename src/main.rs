use burn::tensor::{backend::Backend, Tensor};
use tictactoe::TicTacToe;
mod tictactoe;

fn game<B: Backend>() {
    let game: TicTacToe<B> = TicTacToe::init();
    let test = game.get_initial_state();
    let test2 = game.get_next_state(&test, 8, 1);
    let test3 = game.get_next_state(&test2, 1, -1);
    println!("{:}", test);
    println!("{:}", test2);
    println!("{:}", test3);
}

fn main() {
    game::<burn::backend::CudaJit>();
}
