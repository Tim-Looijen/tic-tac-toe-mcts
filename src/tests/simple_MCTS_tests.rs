#[cfg(test)]
mod MCTS_tests {
    use anyhow::{Ok, Result};
    use ndarray::Array2;
    use test_case::test_case;

    use crate::games::TicTacToe;
    use crate::mcts::Mcts;
    use crate::HashMap;

    #[test_case(vec![(0, 1, 1), (0, 2, 1)] ; "0-0 player one")]
    #[test_case(vec![(0, 0, 1), (0, 2, 1)] ; "0-1 player one")]
    #[test_case(vec![(0, 0, 1), (0, 1, 1)] ; "0-2 player one")]
    //
    #[test_case(vec![(1, 1, 1), (1, 2, 1)] ; "1-0 player one")]
    #[test_case(vec![(1, 0, 1), (1, 2, 1)] ; "1-1 player one")]
    #[test_case(vec![(1, 0, 1), (1, 1, 1)] ; "1-2 player one")]
    //
    #[test_case(vec![(2, 1, 1), (2, 2, 1)] ; "2-0 player one")]
    #[test_case(vec![(2, 0, 1), (2, 2, 1)] ; "2-1 player one")]
    #[test_case(vec![(2, 0, 1), (2, 1, 1)] ; "2-2 player one")]
    //
    #[test_case(vec![(0, 1, -1), (0, 2, -1)] ; "0-0 player two")]
    #[test_case(vec![(0, 0, -1), (0, 2, -1)] ; "0-1 player two")]
    #[test_case(vec![(0, 0, -1), (0, 1, -1)] ; "0-2 player two")]
    //
    #[test_case(vec![(1, 1, -1), (1, 2, -1)] ; "1-0 player two")]
    #[test_case(vec![(1, 0, -1), (1, 2, -1)] ; "1-1 player two")]
    #[test_case(vec![(1, 0, -1), (1, 1, -1)] ; "1-2 player two")]
    //
    #[test_case(vec![(2, 1, -1), (2, 2, -1)] ; "2-0 player two")]
    #[test_case(vec![(2, 0, -1), (2, 2, -1)] ; "2-1 player two")]
    #[test_case(vec![(2, 0, -1), (2, 1, -1)] ; "2-2 player two")]
    fn tictactoe_coords_with_one_winning_move(
        player_coordinates: Vec<(usize, usize, i8)>,
    ) -> Result<()> {
        let game = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        #[allow(clippy::unwrap_used)]
        let player = player_coordinates.first().unwrap().2;
        let state = game.create_state(player_coordinates);

        let mut state = state.clone();
        game.print_state(&state)?;

        let best_action = get_best_action(args.clone(), TicTacToe::init(), &state, player);

        state = game.apply_move(&state, player, best_action);

        game.print_state(&state)?;
        assert!(TicTacToe::init().check_win(&state, player));
        Ok(())
    }

    fn get_best_action(
        args: HashMap<&str, f32>,
        game: TicTacToe,
        given_state: &Array2<i8>,
        player: i8,
    ) -> (usize, usize) {
        let mut tree = Mcts::new(args, game, given_state, player);
        tree.search()
    }
}
