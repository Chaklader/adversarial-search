from minimax_helpers import *
def minimax_decision(gameState):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """
    # TODO: Finish this function!
    pass


# Uncomment the code below to pre-test your helper function
'''
import gamestate as game
best_moves = set([(0, 0), (2, 0), (0, 1)])
rootNode = game.GameState()
minimax_move = minimax_decision(rootNode)

print("Best move choices: {}".format(list(best_moves)))
print("Your code chose: {}".format(minimax_move))

if minimax_move in best_moves:
    print("That's one of the best move choices. Looks like your minimax-decision function worked!")
else:
    print("Uh oh...looks like there may be a problem.")
'''