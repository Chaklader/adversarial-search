from src.minimax.minimax import minimax_decision


class MinimaxPlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.time_left = None
        self.DEPTH_LIMIT = 3  # You can adjust this value

    def get_action(self, state):
        def iterative_deepening():
            best_move = None
            for depth in range(1, self.DEPTH_LIMIT + 1):
                best_move = minimax_decision(state, depth)
            return best_move

        return iterative_deepening()

    # You might need to implement these methods depending on your framework
    def get_name(self):
        return "MinimaxPlayer"

    def set_player_id(self, player_id):
        self.player_id = player_id

    def set_time_left(self, time_left):
        self.time_left = time_left