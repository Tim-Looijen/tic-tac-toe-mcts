#[cfg(test)]
mod MCTS {
    use crate::tictactoe::TicTacToe;
    use crate::HashMap;
    use crate::MCTS::MCTS;

    #[test]
    fn tictactoe_with_1_winning_move_player_one() {
        let game: TicTacToe<burn::backend::Cuda> = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        let mut state = game.get_initial_state();
        state = game.get_next_state(&state, &(1, 0), 1);
        state = game.get_next_state(&state, &(0, 1), -1);
        state = game.get_next_state(&state, &(2, 0), 1);
        state = game.get_next_state(&state, &(0, 2), -1);

        let MCTS_player: i8 = 1;

        TicTacToe::print_state(&state);
        let mut tree = MCTS::new(&game, &args, &state, MCTS_player);
        let best_action = tree.search();
        state = game.get_next_state(&state, &best_action, MCTS_player);

        TicTacToe::print_state(&state);
        assert!(game.check_win(&state, MCTS_player));
    }

    #[test]
    fn tictactoe_with_1_winning_move_player_two() {
        let game: TicTacToe<burn::backend::Cuda> = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        let mut state = game.get_initial_state();
        state = game.get_next_state(&state, &(1, 0), -1);
        state = game.get_next_state(&state, &(0, 1), 1);
        state = game.get_next_state(&state, &(2, 0), -1);
        state = game.get_next_state(&state, &(0, 2), 1);

        let MCTS_player: i8 = -1;

        TicTacToe::print_state(&state);
        let mut tree = MCTS::new(&game, &args, &state, MCTS_player);
        let best_action = tree.search();
        state = game.get_next_state(&state, &best_action, MCTS_player);

        TicTacToe::print_state(&state);
        assert!(game.check_win(&state, MCTS_player));
    }

    #[test]
    fn tictactoe_perfer_draw_over_lose_player_one() {
        let game: TicTacToe<burn::backend::Cuda> = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        let mut state = game.get_initial_state();
        state = game.get_next_state(&state, &(0, 1), -1);
        state = game.get_next_state(&state, &(1, 1), -1);
        state = game.get_next_state(&state, &(2, 2), -1);
        state = game.get_next_state(&state, &(0, 0), 1);
        state = game.get_next_state(&state, &(2, 0), 1);

        let MCTS_player: i8 = 1;

        TicTacToe::print_state(&state);
        let mut tree = MCTS::new(&game, &args, &state, MCTS_player);
        let best_action = tree.search();
        state = game.get_next_state(&state, &best_action, MCTS_player);

        TicTacToe::print_state(&state);
        assert_eq!(best_action, (2, 1));
    }

    #[test]
    fn tictactoe_perfer_draw_over_lose_player_two() {
        let game: TicTacToe<burn::backend::Cuda> = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        let mut state = game.get_initial_state();
        state = game.get_next_state(&state, &(0, 1), 1);
        state = game.get_next_state(&state, &(1, 1), 1);
        state = game.get_next_state(&state, &(2, 2), 1);
        state = game.get_next_state(&state, &(0, 0), -1);
        state = game.get_next_state(&state, &(2, 0), -1);

        let MCTS_player: i8 = -1;

        TicTacToe::print_state(&state);
        let mut tree = MCTS::new(&game, &args, &state, MCTS_player);
        let best_action = tree.search();
        state = game.get_next_state(&state, &best_action, MCTS_player);

        TicTacToe::print_state(&state);
        assert_eq!(best_action, (2, 1));
    }

    #[test]
    fn tictactoe_second_to_last_move_player_one() {
        let game: TicTacToe<burn::backend::Cuda> = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        let mut state = game.get_initial_state();
        state = game.get_next_state(&state, &(0, 0), 1);
        state = game.get_next_state(&state, &(0, 2), 1);
        state = game.get_next_state(&state, &(1, 2), 1);
        state = game.get_next_state(&state, &(2, 1), 1);
        state = game.get_next_state(&state, &(0, 1), -1);
        state = game.get_next_state(&state, &(1, 1), -1);
        state = game.get_next_state(&state, &(2, 2), -1);

        let MCTS_player: i8 = -1;

        TicTacToe::print_state(&state);
        println!("{:?}", game.get_legal_moves(&state));
        let mut tree = MCTS::new(&game, &args, &state, MCTS_player);
        let best_action = tree.search();
        state = game.get_next_state(&state, &best_action, MCTS_player);

        TicTacToe::print_state(&state);
        assert_eq!(best_action, (2, 0));
    }

    #[test]
    fn tictactoe_second_to_last_move_player_two() {
        let game: TicTacToe<burn::backend::Cuda> = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        let mut state = game.get_initial_state();
        state = game.get_next_state(&state, &(0, 0), -1);
        state = game.get_next_state(&state, &(0, 2), -1);
        state = game.get_next_state(&state, &(1, 2), -1);
        state = game.get_next_state(&state, &(2, 1), -1);
        state = game.get_next_state(&state, &(0, 1), 1);
        state = game.get_next_state(&state, &(1, 1), 1);
        state = game.get_next_state(&state, &(2, 2), 1);

        let MCTS_player: i8 = 1;

        TicTacToe::print_state(&state);
        println!("{:?}", game.get_legal_moves(&state));
        let mut tree = MCTS::new(&game, &args, &state, MCTS_player);
        let best_action = tree.search();
        state = game.get_next_state(&state, &best_action, MCTS_player);

        TicTacToe::print_state(&state);
        assert_eq!(best_action, (2, 0));
    }
}
