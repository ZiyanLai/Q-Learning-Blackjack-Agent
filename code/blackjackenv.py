from gymnasium import spaces
from gymnasium.envs.toy_text import blackjack
from gymnasium.envs.toy_text.blackjack import *

class ACTIONS:
	DOUBLE_DOWN = 2
	SURRENDER = 3
	SPLIT = 4

class MyBlackJackEnv(blackjack.BlackjackEnv):
	def __init__(self, render_mode = None, natural=False, sab=False, hitorstand_only=False):
		super().__init__(render_mode, natural, sab)
		self.hitorstand_only = hitorstand_only
		if hitorstand_only:
			self.action_space = spaces.Discrete(2)
		else:
			self.action_space = spaces.Discrete(5)
			self.splitted = False
	
	def reset(self, seed = None, options = None):
		if not self.hitorstand_only:
			self.splitted = False
		return super().reset(seed, options)
	
	def step(self, action):
		assert self.action_space.contains(action)

		if action == ACTIONS.DOUBLE_DOWN:
			terminated = True
			self.player.append(draw_card(self.np_random))
			if is_bust(self.player):
				reward = -2.0
			else:
				while sum_hand(self.dealer) < 17:
					self.dealer.append(draw_card(self.np_random))
				reward = 2 * cmp(score(self.player), score(self.dealer))
				if self.sab and is_natural(self.player) and not is_natural(self.dealer):
					# Player automatically wins. Rules consistent with S&B
					reward = 2.0
				elif (
					not self.sab
					and self.natural
					and is_natural(self.player)
					and reward == 2.0
				):
					# Natural gives extra points, but doesn't autowin. Legacy implementation
					reward = 2 * 1.5
			return self._get_obs(), reward, terminated, False, {}

		elif action == ACTIONS.SURRENDER:
			terminated = True
			reward = -0.5
			return self._get_obs(), reward, terminated, False, {}
		
		else:
			return super().step(action)


if __name__ == "__main__":
	myenv = MyBlackJackEnv()
	myenv.reset()
	myenv.step(2)
