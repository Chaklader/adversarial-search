from isolation import Isolation
import time
import random


class CustomPlayer(DataPlayer):
    """
    Implements an agent using advanced search techniques for the game of Isolation.

    This agent uses iterative deepening with alpha-beta pruning and a transposition
    table to make efficient move decisions within the given time limit.
    """

    def __init__(self, player_id):
        super().__init__(player_id)
        self.transposition_table = {}

    def get_action(self, state):
        """
        Select a move from the available legal moves and return it.

        This method implements iterative deepening with alpha-beta pruning
        and uses a transposition table for move ordering and state caching.

        Parameters:
        -----------
        state : isolation.Isolation
            An instance of the Isolation game state.

        Returns:
        --------
        action : tuple(int, int)
            The selected move as a board coordinate (row, column)
        """

        # Initialize variables
        best_move = None
        depth = 1
        start_time = time.time()

        # Time limit is 150ms by default, we'll use 90% of it to be safe
        time_limit = self.context.get('time_limit', 150) * 0.9 / 1000

        def alpha_beta_search(state, depth, alpha, beta, maximizing_player):
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
            # Check if we've seen this state before
            if state in self.transposition_table:
                return self.transposition_table[state]

            if state.terminal_test():
                return state.utility(self.player_id)

            if depth == 0:
                return self.score(state)

            if maximizing_player:
                value = float('-inf')
                for action in state.actions():
                    value = max(value, alpha_beta_search(state.result(action), depth - 1, alpha, beta, False))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            else:
                value = float('inf')
                for action in state.actions():
                    value = min(value, alpha_beta_search(state.result(action), depth - 1, alpha, beta, True))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

            # Store the evaluated state in the transposition table
            self.transposition_table[state] = value
            return value

        # Iterative deepening
        while time.time() - start_time < time_limit:
            current_best_move = None
            best_score = float('-inf')

            # Randomize action order to avoid bias in equally-good moves
            actions = list(state.actions())
            random.shuffle(actions)

            for action in actions:
                new_state = state.result(action)
                score = alpha_beta_search(new_state, depth - 1, float('-inf'), float('inf'), False)

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

    def score(self, state):
        """
        Calculate the heuristic value of a game state.

        This heuristic considers the difference in the number of moves
        available to each player, as well as the centrality of their positions.

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

        # Consider move difference and add a small bonus for central positions
        move_score = len(own_liberties) - len(opp_liberties)

        # Center positions have coordinates closer to 3 (assuming 7x7 board)
        own_centrality = 7 - (abs(own_loc[0] - 3) + abs(own_loc[1] - 3))
        opp_centrality = 7 - (abs(opp_loc[0] - 3) + abs(opp_loc[1] - 3))
        position_score = (own_centrality - opp_centrality) * 0.1

        return move_score + position_score