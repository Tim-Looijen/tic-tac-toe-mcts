use std::env;

use crate::play_interface::player_vs_mcts;

mod games;
mod mcts;
mod play_interface;
mod tests;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    env::set_var("RUST_LOG", "debug");

    //if let Err(error) = self_play() {
    //   eprintln!("Error: {error:?}");
    //}

    if let Err(error) = player_vs_mcts() {
        eprintln!("Error: {error:?}");
    }
}
