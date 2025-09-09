#[cfg(test)]
mod MCTS_tests {
    use rstest::rstest;
    use std::collections::HashMap;

    use anyhow::{Ok, Result};
    use ndarray::Array2;

    use crate::games::TicTacToe;
    use crate::mcts::Mcts;

    #[rstest]
    #[case::board0(vec![(0, 1, player), (0, 2, player)] )]
    #[case::board1(vec![(0, 0, player), (0, 2, player)] )]
    #[case::board2(vec![(0, 0, player), (0, 1, player)] )]
    //
    #[case::board3(vec![(1, 1, player), (1, 2, player)] )]
    #[case::board4(vec![(1, 0, player), (1, 2, player)] )]
    #[case::board5(vec![(1, 0, player), (1, 1, player)] )]
    //
    #[case::board6(vec![(2, 1, player), (2, 2, player)] )]
    #[case::board7(vec![(2, 0, player), (2, 2, player)] )]
    #[case::board8(vec![(2, 0, player), (2, 1, player)] )]
    //
    fn tictactoe_coords_with_one_winning_move(
        #[values(1, -1)] player: i8,
        #[case] player_coordinates: Vec<(usize, usize, i8)>,
    ) -> Result<()> {
        let game = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        #[allow(clippy::unwrap_used)]
        let state = game.create_state(player_coordinates);

        let mut state = state.clone();
        game.print_state(&state)?;

        let best_action = get_best_action(args.clone(), TicTacToe::init(), &state, player);

        state = game.apply_move(&state, player, best_action);

        game.print_state(&state)?;
        assert!(game.check_win(&state, player));
        Ok(())
    }

    #[rstest]
    #[case::board(vec![(1, 1, -player)] )]
    fn tictactoe_mcts_always_wins_or_draws(
        #[values(1, -1)] player: i8,
        #[case] player_coordinates: Vec<(usize, usize, i8)>,
    ) -> Result<()> {
        let game = TicTacToe::init();
        let args: HashMap<&str, f32> =
            HashMap::from([("C", f32::sqrt(2.0)), ("num_searches", 1000.0)]);

        #[allow(clippy::unwrap_used)]
        let state = game.create_state(player_coordinates);

        let mut num_non_draws = 0;

        for _ in 0..100 {
            let mut state = state.clone();
            let mut rollout_player = player;
            loop {
                let best_action =
                    get_best_action(args.clone(), TicTacToe::init(), &state, rollout_player);
                state = game.apply_move(&state, rollout_player, best_action);

                let (value, terminated) = game.get_value_and_terminated(&state, rollout_player);

                if terminated {
                    if value != 0.5 {
                        num_non_draws += 1;
                    }
                    break;
                }
                rollout_player = -rollout_player;
            }
        }

        println!("{}", num_non_draws);
        assert!(num_non_draws == 0);
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
