use ndarray::Array2;
use std::collections::HashMap;
use std::f32;

use crate::games::TicTacToe;

struct Node {
    state: Array2<i8>,
    player: i8,
    legal_moves: Vec<(usize, usize)>,
    action_taken: Option<(usize, usize)>,

    index: usize,
    parent_index: Option<usize>,
    children_indices: Vec<usize>,

    visit_count: u32,
    value_sum: f32,
}

impl Node {
    pub fn new(
        state: Array2<i8>,
        player: i8,
        legal_moves: Vec<(usize, usize)>,
        action_taken: Option<(usize, usize)>,
        index: usize,
        parent_index: Option<usize>,
    ) -> Self {
        Self {
            state,
            player,
            legal_moves,
            action_taken,
            index,
            parent_index,
            children_indices: vec![],
            visit_count: 0,
            value_sum: 0.0,
        }
    }

    /// Checks if this node has been fully expanded, by checking that there are no more legal moves
    /// and that there are children present
    fn is_fully_expanded(&self) -> bool {
        self.legal_moves.is_empty() && !self.children_indices.is_empty()
    }
}

pub struct Mcts<'a> {
    args: HashMap<&'a str, f32>,
    game: TicTacToe,
    tree: Vec<Node>,
}

impl<'a> Mcts<'a> {
    pub fn new(
        args: HashMap<&'a str, f32>,
        game: TicTacToe,
        root_state: &Array2<i8>,
        player: i8,
    ) -> Mcts<'a> {
        let root = Node::new(
            root_state.clone(),
            player,
            game.get_legal_moves(root_state),
            None,
            0,
            None,
        );
        Mcts {
            args,
            game,
            tree: vec![root],
        }
    }

    pub fn search(&mut self) -> (usize, usize) {
        for _ in 0..self.args["num_searches"] as u32 {
            let mut node_index = self.select(0);
            let node = &self.tree[node_index];

            let (mut value, terminated) =
                self.game.get_value_and_terminated(&node.state, node.player);

            if !terminated {
                node_index = self.expand(node_index);
                value = self.simulate(node_index);
            }

            self.backpropagate(node_index, value);
        }

        self.get_best_action()
    }

    /// Loops through the given nodes children, if any, and returns the child with the best UCB value
    #[allow(non_snake_case)]
    fn select(&self, node_index: usize) -> usize {
        let mut node = &self.tree[node_index];
        while node.is_fully_expanded() {
            let mut best_child_index = node.index;
            let mut best_UCB = f32::MIN;
            for child_index in &node.children_indices {
                let child_index = *child_index;
                let parent = node;
                let child = &self.tree[child_index];
                let ucb = self.calculate_UCB(parent, child);
                if ucb > best_UCB {
                    best_child_index = child_index;
                    best_UCB = ucb;
                }
            }
            node = &self.tree[best_child_index];
        }
        node.index
    }

    /// Adds a new child to the node at the given index by selecting a random action
    /// Also updates the given node's state and legal moves left based on the random chosen action
    fn expand(&mut self, node_index: usize) -> usize {
        let node = &self.tree[node_index];
        let (action_index, action) = self.get_random_action(&node.legal_moves);
        let next_state = self.game.apply_move(&node.state, -node.player, action);
        let new_legal_moves = self.game.get_legal_moves(&next_state);

        let child = Node::new(
            next_state,
            -node.player,
            new_legal_moves,
            Some(action),
            self.tree.len(),
            Some(node_index),
        );

        // Remove so that the random action does not create the same child twice
        self.tree[node_index].legal_moves.remove(action_index);

        // The left over legal moves are those which have not been used up as new a child node
        self.tree[node_index].children_indices.push(child.index);
        self.tree.push(child);

        #[allow(clippy::unwrap_used)]
        self.tree.last().unwrap().index
    }

    /// Simulates a game into future based of the given nodes' state
    /// Returns the result/value of that game at the end, while accounting for the change of perspective.
    fn simulate(&'a self, node_index: usize) -> f32 {
        let node = &self.tree[node_index];
        let (value, terminated) = self.game.get_value_and_terminated(&node.state, node.player);

        // Inverd value because the child is the perspective of the opponent's relative to the parnet
        // If child won, then that would not be good for the parent, so the value is inverted
        if terminated {
            return -value;
        }

        let mut rollout_state = node.state.clone();
        let mut rollout_player = -node.player;

        loop {
            let index_action = self.get_random_action(&self.game.get_legal_moves(&rollout_state));

            rollout_state = self
                .game
                .apply_move(&rollout_state, rollout_player, index_action.1);

            let (mut value, terminated) = self
                .game
                .get_value_and_terminated(&rollout_state, rollout_player);

            if terminated {
                // Sets the value back to parents perspective
                if self.tree[node_index].player == rollout_player && value != 0.5 {
                    value = -value;
                }
                println!("{}", value);

                return value;
            }

            rollout_player *= -1;
        }
    }

    /// Backpropagates the given value to all of the given node's parents
    /// While accounting for the difference in perspectives while going up the tree
    fn backpropagate(&mut self, node_index: usize, value: f32) {
        self.tree[node_index].value_sum += value;
        self.tree[node_index].visit_count += 1;

        let parent_index = self.tree[node_index].parent_index;

        if let Some(valid_parent_index) = parent_index {
            self.backpropagate(valid_parent_index, -value);
        }
    }

    /// Gets the child of the root with the most amount of visits and returns the action taken
    fn get_best_action(&self) -> (usize, usize) {
        let root = &self.tree[0];
        let mut best_child_index = 0;
        let mut highest_visit_count = 0;

        for child_index in &root.children_indices {
            let child_index = *child_index;
            let child = &self.tree[child_index];
            if child.visit_count > highest_visit_count {
                best_child_index = child_index;
                highest_visit_count = child.visit_count;
            }
        }

        #[allow(clippy::unwrap_used)]
        self.tree[best_child_index].action_taken.unwrap()
    }

    /// Choses a random action based on the given node's legal moves left
    fn get_random_action(&self, legal_moves: &[(usize, usize)]) -> (usize, (usize, usize)) {
        let chosen_index = rand::random_range(0..legal_moves.len());
        (chosen_index, legal_moves[chosen_index])
    }

    /// Calculates the UCB for the child, used to determine what 'path' the selection phase should
    /// take in order to get the desired node.
    #[allow(non_snake_case)]
    fn calculate_UCB(&self, parent: &Node, child: &Node) -> f32 {
        let w: f32 = child.value_sum;
        let n: f32 = child.visit_count as f32;
        let N: f32 = parent.visit_count as f32;

        // The constant representing the confidence of the model, the higher this value, the less the win_odds matters in the formula
        // Lower means Exploitation, higher means Exploration
        let C: f32 = self.args["C"];

        let win_odds_child: f32 = if n > 0.0 { w / n } else { 0.0 };

        let q: f32 = win_odds_child;

        let UCB: f32 = q + C * f32::sqrt(N.ln_1p() / n);

        UCB
    }
}
