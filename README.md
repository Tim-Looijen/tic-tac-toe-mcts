# Tic Tac Toe with MCTS

A Tic Tac Toe game written in Rust that plays automatically using a self-written Monte Carlo Tree Search (MCTS).  
Made to learn about how MCTS works, will play against itself until game finish when the program will exit.

## Features
- Simple Tic Tac Toe game using Ndarray
- Self-play using self-written MCTS
- Added tests to verify that the MCTS algorithm chooses the optimal position for different board states

## Running
```bash
cargo run --release
```
## Testing
```bash
cargo test --release
```
