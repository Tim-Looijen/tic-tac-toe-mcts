# Tic Tac Toe with MCTS

A Tic Tac Toe game written in Rust that plays itself using a self-written Monte Carlo Tree Search (MCTS).  
Made to learn about how MCTS works, will play against itself until game finish when the program will exit.
Will always result in a draw when played optimally from both sides.

## Features
- Simple Tic Tac Toe game using Ndarray
- Self-play using self-written MCTS
- Play against the MCTS algorithm using a simple terminal interface
- Added tests to verify that the MCTS algorithm chooses the optimal position for different board states

## Running
```bash
cargo run --release
```
## Testing
```bash
cargo test --release
```
