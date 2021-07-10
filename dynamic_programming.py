import random

import numpy as np
from kaggle_environments import utils

import utils

# def agent_1(obs, config):
#     state = (tuple(obs.board), obs.mark, obs.mark)
#     actions = get_available_actions(state)
#     winning_actions = []
#     for action in actions:
#         board, mark, player = get_next_state(state, action)
#         if check_winning_board(board, player):
#             winning_actions.append(action)
#
#     if len(winning_actions) > 0:
#         return random.choice(winning_actions)
#
#     return random.choice(actions)
#
#
# def agent_2(obs, config):
#     state = (tuple(obs.board), obs.mark, obs.mark)
#     actions = get_available_actions(state)
#
#     max_reward = -200
#     best_action = None
#
#     for action in actions:
#         next_state, reward = get_next_state_and_reward(state, action)
#         if reward > max_reward:
#             max_reward = reward
#             best_action = action
#     return best_action


def compute_pi(theta=0.001, gamma=0.9):
    states = utils.load_state_file('tictactoe_states.txt')
    V = {state: 0 for state in states}
    PI = {}
    for state in states:
        actions = utils.get_available_actions(state)
        PI[state] = random.choice(actions) if len(actions) != 0 else None

    while True:
        while True:
            delta = 0

            for state in states:
                v = V[state]

                board, mark, player = state

                opp = 3 - player
                opp_state = (board, mark, opp)

                if PI[state] is not None:
                    next_state, reward = utils.get_next_state_and_reward(state, PI[state])
                    V[state] = reward + gamma * V[next_state]
                elif PI[opp_state] is not None:
                    opp_next_state, opp_reward = utils.get_next_state_and_reward(opp_state, PI[opp_state])

                    board, mark, _ = opp_next_state
                    next_state = (board, mark, player)

                    V[state] = -opp_reward + gamma * V[next_state]

                delta = max(delta, abs(v - V[state]))

            if delta < theta:
                break

        policy_stable = True
        for state in states:
            old_action = PI[state]

            actions = utils.get_available_actions(state)

            max_return = -np.inf
            best_action = None

            for action in actions:
                next_state, reward = utils.get_next_state_and_reward(state, action)

                return_value = reward + gamma * V[state]
                if return_value > max_return:
                    max_return = return_value
                    best_action = action

            PI[state] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break

    return PI, V
