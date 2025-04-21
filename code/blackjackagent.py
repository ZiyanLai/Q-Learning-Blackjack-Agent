from collections import defaultdict
from functools import partial
from blackjackenv import ACTIONS
import numpy as np
import random as rand

class BlackjackBasicAgent:
        
    def get_optimal_action(self, env, state):
        player_sum = state[0]
        dealer_face = state[1]
        if len(env.player) == 2:
            if (player_sum == 10 or player_sum == 11) and dealer_face <= 9:
                return ACTIONS.DOUBLE_DOWN
            
        if player_sum < 12:
            return 1
        elif 12 <= player_sum <= 16:
            if dealer_face >= 7 or dealer_face == 1:
                return 1
            else:
                return 0
        else:
            return 0

class BlackjackAgent:
    def __init__(
        self,
        env, 
        learning_rate = 0.01,
        random_action_rate = 1,
        random_action_rate_decay = 0.99,
        min_random_action_rate = 0.01,
        discount_factor = 0.95
    ):
        # self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.q_table = defaultdict(partial(np.zeros, env.action_space.n))
        self.alpha = learning_rate
        self.rar = random_action_rate
        self.min_rar = min_random_action_rate
        self.rar_decay = random_action_rate_decay
        self.discount = discount_factor
        
    def get_action(self, env, state):
        if env.hitorstand_only:
            max_action = 1
        else:
            if len(env.player) > 2:
                max_action = 1
            elif env.player[0] == env.player[1] and env.splitted == False:
                max_action = 4
            else:
                max_action = 3
            
        if rand.uniform(0, 1) <= self.rar:
            # action = env.action_space.sample()
            action = rand.choice(range(0, max_action+1))
        else:
            action = self.get_optimal_action(env, state, max_action=max_action)
    
        return action

    def get_optimal_action(self, env, state, max_action=4):
        return int(np.argmax(self.q_table[state][0:max_action+1]))
    
    def update(self, state, next_state, action, reward, terminated):
        if terminated:
            next_q_val = 0
        else:
            next_q_val = max(self.q_table[next_state])
            
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                                    self.alpha * (reward + self.discount * next_q_val)

    def decay_rar(self):
        self.rar = max(self.min_rar, self.rar * self.rar_decay)