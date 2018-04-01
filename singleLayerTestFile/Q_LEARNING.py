import numpy as np
import pandas as pd
import time


np.random.seed(2)

n_STATE = 6
ACTIONS= ['left', 'right']
EPSILION = 0.95
LEARNING_RATE = 0.3
DISCOUNT = 0.9
MAX_EPSILION = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),
                         columns=actions)
    print(table)
    return(table)

def build_reward_table(n_states):
    table = pd.DataFrame(np.zeros((n_states, n_states)))
    table.iloc[4, 5] = 1
    table.iloc[5, 5] = 1
    print(table)
    return(table)


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILION) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_feedback(state, action):
    if action == 'right':
        if state == n_STATE - 2:
            state = n_STATE - 1
        else:
            state += 1
    elif state != 0:
            state -= 1
    return state

def env_update(state, episode, step_counter):
    env_list = ['_']*(n_STATE-1) + ['T']
    if state == n_STATE - 1:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                             ', end='')
    else:
        env_list[state] = 'H'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(n_STATE, ACTIONS)
    reward_table = build_reward_table(n_STATE)
    for episode in range(MAX_EPSILION):
        step_counter = 0
        state = 0
        is_terminated = False
        env_update(state, episode, step_counter)
        while not is_terminated:
            curr_action = choose_action(state, q_table)
            state_new = get_feedback(state, curr_action)
            curr_reword = reward_table.iloc[state, state_new]
            q_table.loc[[state], [curr_action]] = (1-LEARNING_RATE)*q_table.loc[[state], [curr_action]] + LEARNING_RATE * (curr_reword + DISCOUNT * q_table.loc[state_new,:].max())
            state = state_new
            if state == n_STATE - 1:
                is_terminated = True
            step_counter += 1
            env_update(state, episode, step_counter)
            print('\n')
        print(q_table)
    return q_table



rl()















