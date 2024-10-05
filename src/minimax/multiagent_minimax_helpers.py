def min_value(gameState):
    """ Return the game state utility if the game is over,
    otherwise return the minimum value over all legal successors
    
    # HINT: Assume that the utility is ALWAYS calculated for
            player 1, NOT for the "active" player
    """
    # TODO: finish this function!
    pass


def max_value(gameState):
    """ Return the game state utility if the game is over,
    otherwise return the maximum value over all legal successors
    
    # HINT: Assume that the utility is ALWAYS calculated for
            player 1, NOT for the "active" player
    """
    # TODO: finish this function!
    pass

# Uncomment the code below to pre-test your helper function
'''
from gamestate import *
g = GameState()
inf = float("inf")
actions = [((0, 0), inf), ((1, 0), -inf), ((2, 0), inf), ((0, 1), inf), ((1, 1), -inf)]

if all(min_value(g.result(a)) == ev for a, ev in actions):
    print("Looks like everything works!")
else:
    print("Uh oh! Not all the scores matched.")
'''