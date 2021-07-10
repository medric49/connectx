import string

import numpy as np
from kaggle_environments import make, utils


def create_state_file(filename):
    env = make('tictactoe', debug=True)
    states = set()
    for i in range(10000):
        for a in env.run(['random', 'random']):
            s = tuple(a[0]['observation']['board'])
            states.add(s)

    for i in range(10000):
        for a in env.run(['reaction', 'random']):
            s = tuple(a[0]['observation']['board'])
            states.add(s)

    for i in range(10000):
        for a in env.run(['random', 'reaction']):
            s = tuple(a[0]['observation']['board'])
            states.add(s)

    state_file = open(filename, 'w')
    for s in states:
        state_file.write(f'{s}\n')
    state_file.close()


def load_state_file(filename):
    state_file = open(filename, 'r')
    states = []
    for s in state_file:
        state = []
        for c in s:
            if c in string.digits:
                state.append(int(c))
        s = tuple(state)
        if state.count(1) == state.count(2):
            states.append((s, 1, 1))
            states.append((s, 1, 2))
        elif state.count(1) > state.count(2):
            states.append((s, 2, 1))
            states.append((s, 2, 2))
    state_file.close()
    return states


def make_grid(board):
    grid = np.asarray(list(board)).reshape(3, 3)
    return grid


def check_winning_board(board, player):
    grid = make_grid(board)
    for r in range(3):
        window = list(grid[r, :])

        if window.count(player) == 3:
            return True
    for c in range(3):
        window = list(grid[:, c])
        if window.count(player) == 3:
            return True

    window = []
    for i in range(3):
        window.append(grid[i, i])
    if window.count(player) == 3:
        return True

    window = []
    for i in range(3):
        window.append(grid[2 - i, i])
    if window.count(player) == 3:
        return True

    return False


def check_losing_board(board, player):
    return check_winning_board(board, 3 - player)


def check_defending_board(board, player):
    grid = make_grid(board)
    opponent = 3 - player
    for r in range(3):
        window = list(grid[r, :])

        if window.count(opponent) == 2 and window.count(player) == 1:
            return True

    for c in range(3):
        window = list(grid[:, c])
        if window.count(opponent) == 2 and window.count(player) == 1:
            return True

    window = []
    for i in range(3):
        window.append(grid[i, i])
    if window.count(opponent) == 2 and window.count(player) == 1:
        return True

    window = []
    for i in range(3):
        window.append(grid[2 - i, i])
    if window.count(opponent) == 2 and window.count(player) == 1:
        return True

    return False


def check_almost_winning_board(board, player):
    grid = make_grid(board)
    opponent = 3 - player
    for r in range(3):
        window = list(grid[r, :])

        if window.count(player) == 2 and window.count(opponent) == 0:
            return True

    for c in range(3):
        window = list(grid[:, c])
        if window.count(player) == 2 and window.count(opponent) == 0:
            return True

    window = []
    for i in range(3):
        window.append(grid[i, i])
    if window.count(player) == 2 and window.count(opponent) == 0:
        return True

    window = []
    for i in range(3):
        window.append(grid[2 - i, i])
    if window.count(player) == 2 and window.count(opponent) == 0:
        return True

    return False


def finished_board(board):
    if board.count(0) == 0:
        return True

    grid = make_grid(board)
    for r in range(3):
        window = list(grid[r, :])

        if window.count(1) == 3 or window.count(2) == 3:
            return True
    for c in range(3):
        window = list(grid[:, c])
        if window.count(1) == 3 or window.count(2) == 3:
            return True

    window = []
    for i in range(3):
        window.append(grid[i, i])
    if window.count(1) == 3 or window.count(2) == 3:
        return True

    window = []
    for i in range(3):
        window.append(grid[2 - i, i])
    if window.count(1) == 3 or window.count(2) == 3:
        return True

    return False


def get_final_states(filename):
    all_states = load_state_file(filename)
    states = set()
    for state in all_states:
        board, mark, player = state
        if finished_board(board):
            states.add(state)
    return states


def get_available_actions(state):
    board, mark, player = state
    grid = make_grid(board)

    if mark != player:
        return []

    if finished_board(board):
        return []

    actions = []
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:
                actions.append(i*3+j)
    return actions


def get_next_state(state, action):
    board, mark, player = state

    next_board = list(board).copy()
    next_board[action] = mark

    return tuple(next_board), 3 - mark, player


def get_next_state_and_reward(state, action):
    next_state = get_next_state(state, action)
    reward = get_reward(next_state)
    return next_state, reward


def get_reward(next_state):
    next_board, next_mark, player = next_state

    reward = 0
    if check_winning_board(next_board, player):
        reward += 75
    if check_winning_board(next_board, 3 - player):
        reward += -100
    if check_almost_winning_board(next_board, player):
        reward += 10
    if check_almost_winning_board(next_board, 3 - player):
        reward += -25
    if check_defending_board(next_board, player):
        reward += 15
    if next_board.count(0) == 0:
        reward += 25

    return reward
