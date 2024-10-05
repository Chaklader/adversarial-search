import random
import time

from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """
    Implements an agent using advanced search techniques for the game of Isolation.

    This agent uses iterative deepening with alpha-beta pruning to make
    efficient move decisions within the given time limit.
    """

    def get_action(self, state):
        """
        Select a move from the available legal moves and return it.

        This method implements iterative deepening with alpha-beta pruning.
        It ensures that at least one action is always available before timeout.

        Parameters:
        -----------
        state : isolation.Isolation
            An instance of the Isolation game state.

        Returns:
        --------
        action : tuple(int, int)
            The selected move as a board coordinate (row, column)
        """

        # Immediately put a random action in the queue to ensure we always have a move
        if state.actions():
            self.queue.put(random.choice(state.actions()))

        # Initialize variables
        best_move = None
        depth = 1

        # Use a fixed time limit if self.context is None
        time_limit = 150 * 0.9 / 1000  # 135ms in seconds

        # Only use self.context if it's not None and has the 'get' attribute
        if self.context is not None and hasattr(self.context, 'get'):
            time_limit = self.context.get('time_limit', 150) * 0.9 / 1000

        start_time = time.time()

        # Iterative deepening
        while time.time() - start_time < time_limit:
            current_best_move = None
            best_score = float('-inf')

            # Randomize action order to avoid bias in equally-good moves
            actions = list(state.actions())
            random.shuffle(actions)

            for action in actions:
                new_state = state.result(action)
                score = self.alpha_beta_search(new_state, depth - 1, float('-inf'), float('inf'), False)

                if score > best_score:
                    best_score = score
                    current_best_move = action

            # Update the overall best move
            if current_best_move:
                best_move = current_best_move
                # Always have a move ready to return
                self.queue.put(best_move)

            # Increase depth for next iteration
            depth += 1

        return best_move

    def alpha_beta_search(self, state, depth, alpha, beta, maximizing_player):
        """
        Perform alpha-beta pruning at a given depth.

        Parameters:
        -----------
        state : isolation.Isolation
            The current game state
        depth : int
            The current depth in the game tree
        alpha : float
            The alpha value for pruning
        beta : float
            The beta value for pruning
        maximizing_player : bool
            True if the current player is maximizing, False otherwise

        Returns:
        --------
        float
            The evaluated score for the current state
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth == 0:
            return self.score(state)

        if maximizing_player:
            value = float('-inf')
            for action in state.actions():
                value = max(value, self.alpha_beta_search(state.result(action), depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for action in state.actions():
                value = min(value, self.alpha_beta_search(state.result(action), depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def score(self, state):
        """
        Calculate the heuristic value of a game state.

        This heuristic considers the difference in the number of moves
        available to each player.

        Parameters:
        -----------
        state : isolation.Isolation
            The game state to evaluate

        Returns:
        --------
        float
            The heuristic value of the state
        """
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        return len(own_liberties) - len(opp_liberties)