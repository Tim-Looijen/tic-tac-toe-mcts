use std::{
    collections::HashMap,
    io::{stdin, stdout, Write},
    usize,
};

use anyhow::Result;
use ndarray::Array2;

use crate::{games::TicTacToe, mcts::Mcts};

pub fn choose_play_option() -> Result<()> {
    loop {
        print!("Which player, X/x or O/o: ");
        // Choose mcts self play, or play against mcts
        // On terminated, chose again
        // provide way to exit loop
    }

    Ok(())
}

fn get_input() -> Result<String> {
    let mut input = String::new();

    let _ = stdout().flush();
    stdin().read_line(&mut input)?;

    if let Some('\n') = input.chars().next_back() {
        input.pop();
    }

    if let Some('\r') = input.chars().next_back() {
        input.pop();
    }

    Ok(input)
}

pub fn self_play() -> Result<()> {
    let game = TicTacToe::init();
    let args: HashMap<&str, f32> =
        HashMap::from([(("C"), f32::sqrt(2.0)), (("num_searches"), 1000.0)]);

    let mut state = game.get_initial_state();
    let mut player = 1;

    game.print_state(&state)?;
    loop {
        let mut tree = Mcts::new(args.clone(), TicTacToe::init(), &state, player);
        let best_action = tree.search();
        state = game.apply_move(&state, player, best_action);
        game.print_state(&state)?;

        if game.get_value_and_terminated(&state, player).1 {
            break;
        }

        player = -player;
    }
    Ok(())
}

pub fn player_vs_mcts() -> Result<()> {
    print!("Which player, X/x or O/o: ");
    let mut chosen_player: i8;
    loop {
        let player_input = get_input()?;

        chosen_player = match player_input.to_lowercase().trim() {
            "X" | "x" => 1,
            "O" | "o" => -1,
            _ => 0,
        };

        if chosen_player == 0 {
            print!(
                "Invalid player: \"{}\", please choose one of these: 'X'/'x' or 'O'/'o': ",
                player_input
            );
        } else {
            let chosen_player_as_char = if chosen_player == 1 { "X" } else { "O" };

            println!("Player \"{}\" chosen!", chosen_player_as_char);
            break;
        }
    }

    let mcts_player = -chosen_player;
    let game = TicTacToe::init();
    let mut state = game.get_initial_state();

    if mcts_player == 1 {
        state = mcts_turn(&game, &state, mcts_player)?;
    }

    game.print_state(&state)?;
    loop {
        state = player_turn(&game, &state, chosen_player)?;
        let (value, terminated) = game.get_value_and_terminated(&state, chosen_player);

        if terminated {
            // Either draw or player won
            break;
        }

        state = mcts_turn(&game, &state, mcts_player)?;
        let (value, terminated) = game.get_value_and_terminated(&state, mcts_player);

        if terminated {
            // Either draw or MCTS won
            break;
        }
    }

    Ok(())
}

fn mcts_turn(game: &TicTacToe, state: &Array2<i8>, mcts_player: i8) -> Result<Array2<i8>> {
    let player_as_char = if mcts_player == 1 { "X" } else { "O" };
    print!("MCTS turn, playing as '{}':", player_as_char);

    let args: HashMap<&str, f32> =
        HashMap::from([(("C"), f32::sqrt(2.0)), (("num_searches"), 1000.0)]);
    let mut tree = Mcts::new(args.clone(), TicTacToe::init(), state, mcts_player);
    let state = game.apply_move(state, mcts_player, tree.search());
    game.print_state(&state)?;
    Ok(state)
}

fn player_turn(game: &TicTacToe, state: &Array2<i8>, chosen_player: i8) -> Result<Array2<i8>> {
    let chosen_player_as_char = if chosen_player == 1 { "X" } else { "O" };

    println!("Valid options: {:?}", game.get_legal_moves(state));
    print!(
        "Where do you want to put the {}? (Enter the row and then the column, such as '01')",
        chosen_player_as_char
    );

    let chosen_action = loop {
        let input = get_input()?;

        if let [chosen_row, chosen_column] = input.chars().collect::<Vec<_>>()[..] {
            if let (Some(row), Some(col)) = (chosen_row.to_digit(10), chosen_column.to_digit(10)) {
                if row <= 2 && col <= 2 {
                    break (row as usize, col as usize);
                }
            }
        }

        print!(
            "Invalid syntax (\"{}\"), please only provide 2 numbers, between 0 and 2: ",
            input
        );
    };

    print!(
        "Chosen location for '{}': {:?}:",
        chosen_player_as_char, chosen_action
    );

    let state = game.apply_move(state, chosen_player, chosen_action);
    game.print_state(&state)?;

    Ok(state.clone())
}
