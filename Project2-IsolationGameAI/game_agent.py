"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def weighted_my_moves(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Modify the Open moves score heuristic provided to us by weighting each
    open move by a weight that depends on the position the move leads to
    on the game board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    h, w = game.height, game.width
    score = 0
    own_moves = game.get_legal_moves(player)
    for move in own_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            score += 6
        else:
            score += 8

    return float(score)

def weighted_diff_my_moves_opp_moves(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Calculate the difference in the weighted open moves scores between the
    current player and its opponent and use that as the score of the
    current game state.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    h, w = game.height, game.width
    own_score = 0
    opp_score = 0
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    for move in own_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            own_score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            own_score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            own_score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            own_score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            own_score += 6
        else:
            own_score += 8

    for move in opp_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            opp_score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            opp_score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            opp_score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            opp_score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            opp_score += 6
        else:
            opp_score += 8

    #print(own_score, opp_score, own_score - opp_score)
    return float(own_score - opp_score)

def diff_my_moves_opp_moves_one_ply_lookahead(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    The difference in the number of available moves between the current
    player and its opponent one ply ahead in the future is used as the
    score of the current game state.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : objects
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    h, w = game.height, game.width
    own_score = 0
    opp_score = 0
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    for move in own_moves:
        own_score += len(game.__get_moves__(move))

    for move in opp_moves:
        opp_score += len(game.__get_moves__(move))

    #print(own_score, opp_score, own_score - opp_score)
    return float(own_score - opp_score)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    return diff_my_moves_opp_moves_one_ply_lookahead(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) == 0:
            return (-1, -1)
        elif len(legal_moves) == 1:
            return legal_moves[0]

        h, w = game.height, game.width
        # Opening Book Rules
        # First move of first player (current player)
        if game.move_count == 0:
            return ((h-1)//2, (w-1)//2)

        # First move of second player (current player)
        if game.move_count == 1:
            opp_location = game.get_player_location(game.get_opponent(self))
            if opp_location not in game.__get_moves__(((h-1)//2, (w-1)//2)):
                if game.move_is_legal(((h-1)//2, (w-1)//2)):
                    return ((h-1)//2, (w-1)//2)
                else:
                    return ((h-1)//2 - 1, (w-1)//2)
            else:
                return ((h-1)//2 - 1, (w-1)//2)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            best_move = random.choice(legal_moves)
            # Iterative Deepening Search
            if self.iterative:
                if self.method == 'minimax':
                    d = 1
                    while True:
                        branch_score, best_move = self.minimax(game, depth=d)
                        # print("Current Depth: ", d, "Score: ", branch_score, "Best Move: ", best_move)
                        d += 1
                elif self.method == 'alphabeta':
                    d = 1
                    while True:
                        branch_score, best_move = self.alphabeta(game, depth=d)
                        # print("Current Depth: ", d, "Score: ", branch_score, "Best Move: ", best_move)
                        d += 1
                else:
                    raise "Invalid method name!"
            else:
                if self.method == 'minimax':
                    branch_score, best_move = self.minimax(game,
                                                depth=self.search_depth)
                elif self.method == 'alphabeta':
                    branch_score, best_move = self.alphabeta(game,
                                                depth=self.search_depth)
                else:
                    raise "Invalid method name!"
        except Timeout:
            # Handle any actions required at timeout, if necessary
            # print(best_move)

            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get all the available moves at the current state
        available_moves = game.get_legal_moves()

        # Recursion Stopping Conditions
        # Terminal/Leaf Node --> No more legal moves available OR
        # Fixed depth search --> Current depth level exceed specified max depth.
        if (not available_moves) or (depth == 0):
            if maximizing_player:
                return (self.score(game, game.active_player), (-1, -1))
            else:
                return (self.score(game, game.inactive_player), (-1, -1))

        if maximizing_player:
            current_score = float("-inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.minimax(game_child, depth-1,
                                                       False)

                # Identify the maximum score branch for the current player.
                if child_score >= current_score:
                    current_move = move
                    current_score = child_score

        else:
            current_score = float("inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.minimax(game_child, depth-1,
                                                       True)

                # Identify the minimum score branch for the opponent.
                if child_score <= current_score:
                    current_move = move
                    current_score = child_score

        return current_score, current_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.s

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get all the available moves at the current state
        available_moves = game.get_legal_moves()

        # Recursion Stopping Conditions
        # Terminal/Leaf Node --> No more legal moves available OR
        # Fixed depth search --> Current depth level exceed specified max depth.
        if (not available_moves) or (depth == 0):
            if maximizing_player:
                return (self.score(game, game.active_player), (-1, -1))
            else:
                return (self.score(game, game.inactive_player), (-1, -1))

        current_move = available_moves[0]
        if maximizing_player:
            current_score = float("-inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.alphabeta(game_child, depth-1,
                                                           alpha, beta,
                                                           False)

                if child_score >= current_score:
                    current_score = child_score

                # Test if the branch utility is greater than beta,
                # then prune other sibling branches.
                if current_score >= beta:
                    return current_score, move

                # Update alpha if branch utility is greater than
                # current value of alpha for MAX nodes.
                if current_score > alpha:
                    current_move = move
                    alpha = current_score

        else:
            current_score = float("inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.alphabeta(game_child, depth-1,
                                                           alpha, beta,
                                                           True)

                if child_score <= current_score:
                    current_score = child_score

                # Test if the branch utility is less than alpha,
                # then prune other sibling branches.
                if current_score <= alpha:
                    return current_score, move

                # Update beta if branch utility is lesser than
                # current value of beta for MIN nodes.
                if current_score < beta:
                    current_move = move
                    beta = current_score

        return current_score, current_move
