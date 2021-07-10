import utils
import random
import numpy as np
EMPTY_BOARD = (0, 0, 0, 0, 0, 0, 0, 0, 0)


def get_initial_state(player=None):
    board = EMPTY_BOARD
    mark = 1
    if player is None:
        player = random.choice([1, 2])
    return board, mark, player


def best_action_value(Q, state):
    action_values = Q[state]

    actions = []
    value = -np.inf

    for a, v in action_values:
        if v > value:
            actions = [a]
            value = v
        elif v == value:
            actions.append(a)
    return random.choice(actions), value


def fill_q(Q, state):
    board, mark, player = state

    if mark == player:
        actions = utils.get_available_actions(state)
        Q[state] = {action: 0 for action in actions}
    else:
        Q[state] = {None: 0}


def convergence_rate(N_delta):
    N_delta = np.array(list(N_delta.values()))
    N_delta[:, 0] = N_delta[:, 0] / N_delta[:, 0].max()
    rates = N_delta.min(axis=1)
    rate = rates.max()
    return rate




def compute_q(theta=0.001, gamma=0.9):
    Q = {}
    N_delta = {}

    final_states = utils.get_final_states('tictactoe_states.txt')

    state_01 = get_initial_state(player=1)
    fill_q(Q, state_01)

    state_02 = get_initial_state(player=2)
    fill_q(Q, state_02)

    state = get_initial_state()

    while len(N_delta) == 0 or convergence_rate(N_delta) > theta:
        board, mark, player = state

        if mark == player:
            action = random.choice(utils.get_available_actions(state))
            next_state = utils.get_next_state(state, action)
        else:
            action = None
            opp_state = board, mark, 3 - player
            opp_action = random.choice(utils.get_available_actions(opp_state))

            board, mark, opp = utils.get_next_state(opp_state, opp_action)
            next_state = board, mark, player

        if next_state not in Q:
            fill_q(Q, next_state)

        if next_state in final_states:
            target = utils.get_reward(next_state)
            next_state = get_initial_state()
        else:
            target = utils.get_reward(next_state) + gamma * max(Q[next_state].values())

        tmp = Q[state][action]

        if (state, action) not in N_delta:
            N_delta[(state, action)] = [0, None]
        N_delta[(state, action)][0] += 1
        alpha = 1/N_delta[(state, action)][0]

        Q[state][action] = (1 - alpha) * Q[state][action] + alpha * target

        if N_delta[(state, action)][1] is not None:
            N_delta[(state, action)][1] = abs(Q[state][action] - tmp)
        else:
            N_delta[(state, action)][1] = 10

        state = next_state
    return Q
