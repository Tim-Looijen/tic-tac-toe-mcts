use crate::play_interface::choose_play_option;

mod games;
mod mcts;
mod play_interface;
mod tests;

fn main() {
    if let Err(error) = choose_play_option() {
        eprintln!("Error: {error:?}");
    }
}
