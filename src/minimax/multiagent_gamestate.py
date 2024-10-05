from copy import deepcopy

xlim, ylim = 3, 2  # board dimensions

# The eight movement directions possible for a chess queen
RAYS = [(1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1)]



class GameState:
    """
    Attributes
    ----------
    TODO: Copy in your implementation from the previous quiz
    """
    def __init__(self):
        # TODO: Copy in your implementation from the previous quiz
        pass
        
    def actions(self):
        """ Return a list of legal actions for the active player 
        
        You are free to choose any convention to represent actions,
        but one option is to represent actions by the (row, column)
        of the endpoint for the token. For example, if your token is
        in (0, 0), and your opponent is in (1, 0) then the legal
        actions could be encoded as (0, 1) and (0, 2).
        """
        # TODO: Finish this function!
        pass
    
    def player(self):
        """ Return the id of the active player 
        
        Hint: return 0 for the first player, and 1 for the second player
        """
        # TODO: Finish this function!
        pass
    
    def result(self, action):
        """ Return a new state that results from applying the given
        action in the current state
        
        Hint: Check out the deepcopy module--do NOT modify the
        objects internal state in place
        """
        # TODO: Finish this function!
        pass
    
    def terminal_test(self):
        """ return True if the current state is terminal,
        and False otherwise
        
        Hint: an Isolation state is terminal if _either_
        player has no remaining liberties (even if the
        player is not active in the current state)
        """
        # TODO: Finish this function!
        pass
    
    def liberties(self, loc):
        """ Return a list of all open cells in the
        neighborhood of the specified location.  The list 
        should include all open spaces in a straight line
        along any row, column or diagonal from the current
        position. (Tokens CANNOT move through obstacles
        or blocked squares in queens Isolation.)
        
        Note: if loc is None, then return all empty cells
        on the board
        """
        # TODO: Finish this function!
        pass