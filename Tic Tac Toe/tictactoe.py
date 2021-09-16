"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None
board_size = 3


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # initialize count
    xCount = 0
    oCount = 0

    # loop through position on board to count number of X and O played
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == X:
                xCount += 1
            elif board[i][j] == O:
                oCount += 1

    # X plays if empty board
    if xCount == 0 and oCount == 0:
        return X
    # O plays if there are more X than O on board
    elif xCount > oCount:
        return O
    # X plays in other cases (same number of X and O on board)
    else:
        return X
    raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # initialize action set
    actionSet = set()

    # loop through positions on board to find empty position,
    # then add to possible action set
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == EMPTY:
                actionSet.add((i, j))

    return actionSet
    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # if position of action is not empty, raise error
    if board[action[0]][action[1]] != EMPTY:
        raise NameError('Invalid Move')

    newBoard = copy.deepcopy(board)

    # find player to make next move, then add their move to the board
    currPlayer = player(board)
    newBoard[action[0]][action[1]] = currPlayer

    return newBoard
    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check rows for winners
    for i in range(board_size):
        if board[i][0] == board[i][1] and board[i][0] == board[i][2] \
                and board[i][2] != EMPTY:
            return board[i][0]

    # check columns for winners
    for j in range(board_size):
        if board[0][j] == board[1][j] and board[0][j] == board[2][j] \
                and board[2][j] != EMPTY:
            return board[0][j]

    # check diagonals for winners
    if board[0][0] == board[1][1] and board[1][1] == board[2][2] \
            and board[2][2] != EMPTY:
        return board[1][1]
    if board[2][0] == board[1][1] and board[1][1] == board[0][2] \
            and board[0][2] != EMPTY:
        return board[1][1]

    # return None if no winners found
    return None
    raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # if there is winner return True
    if winner(board) is not None:
        return True

    # loop through position, if board is not full return False
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == EMPTY:
                return False

    # return True if all board positions are filled
    return True
    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board) is X:
        return 1
    elif winner(board) is O:
        return -1
    else:
        return 0

    raise NotImplementedError


def maxValue(board, vMax):
    """
    Calculate the max utility achievable with a board state
    """
    # if board reaches terminal state, return utility
    if terminal(board):
        return utility(board)

    # initialize utility variable
    v = -math.inf

    # loop through all possible moves to find move with maximum utility
    for move in actions(board):
        v = max(v, minValue(result(board, move), v))

        # if utility of move is larger than vMax, break out of loop
        # (alpha-beta pruning)
        if v > vMax:
           break
    return v


def minValue(board, vMin):
    """
    Calculate the min utility achievable with a board state
    """
    # if board reaches terminal state, return utility
    if terminal(board):
        return utility(board)

    # initialize utility variable
    v = math.inf

    # loop through all possible moves to find move with minimum utility
    for move in actions(board):
        v = min(v, maxValue(result(board, move), v))

        # if utility of move is less than vMin, break out of loop
        # (alpha-beta pruning)
        if v < vMin:
            break
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # return None if board is terminal
    if terminal(board):
        return None

    # initialize optimal Action variable
    optimalAction = tuple()

    # populate action list given board state
    actionList = actions(board)

    # if player is X, optimal move would be one with max utility
    if player(board) == X:
        v = -math.inf
        # loop through action list to find move with max utility
        for move in actionList:
            moveUtility = minValue(result(board, move), v)
            if moveUtility > v:
                v = moveUtility
                optimalAction = move

            # 1 is maximum utility possible, return optimal move immediately
            if v == 1:
                return optimalAction

    # if player is O, optimal move would be one with min utility
    if player(board) == O:
        v = +math.inf

        # loop through action list to find move with min utility
        for move in actionList:
            moveUtility = maxValue(result(board, move), v)
            if moveUtility < v:
                v = moveUtility
                optimalAction = move

            # -1 is minimum utility possible, return optimal move immediately
            if v == -1:
                return optimalAction

    return optimalAction
    raise NotImplementedError
