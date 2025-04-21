import pickle
import pandas as pd
from gymnasium.envs.toy_text.blackjack import *
from blackjackagent import BlackjackAgent 
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from blackjackenv import ACTIONS
from pathlib import Path

class Mode:
    TRAIN = "train"
    TEST = "test"

def play_one_hand(env, agent, mode=Mode.TRAIN, seed=None):
    state, info = env.reset(seed=seed)
    if mode == Mode.TRAIN:
        action = agent.get_action(env, state)
    elif mode == Mode.TEST:
        action = agent.get_optimal_action(env, state)
    
    if action != ACTIONS.SPLIT:
        done = False
        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            if mode == Mode.TRAIN:
                agent.update(state, next_state, action, reward, terminated)
                state = next_state
                action = agent.get_action(env, state)
                
            elif mode == Mode.TEST:
                state = next_state
                action = agent.get_optimal_action(env, state)

            done = terminated or truncated

        return reward
    
    elif action == ACTIONS.SPLIT:
        env.splitted = True
        card = env.player[0]
        start_state = state
        total_reward = 0

        for _ in range(2):
            done = False
            env.player = [card, draw_card(env.np_random)]
            state, _ = env._get_obs(), {}
            while not done:
                if mode == Mode.TRAIN:
                    action = agent.get_action(env, state)
                elif mode == Mode.TEST:
                    action = agent.get_optimal_action(env, state)

                next_state, reward, terminated, truncated, info = env.step(action)
                if mode == Mode.TRAIN:
                    agent.update(state, next_state, action, reward, terminated)
                done = terminated or truncated
                state = next_state
            
            total_reward += reward
            
        if mode == Mode.TRAIN:     
            agent.update(start_state, None, ACTIONS.SPLIT, total_reward, True)

        return total_reward

def get_or_train(filename, train_fn, *build_args, overwrite=False, **build_kwargs):
	root    = Path(__file__).resolve().parent
	out_dir = root / "pkl"                            
    # out_dir = Path("pkl")
    # out_dir.mkdir(exist_ok=True)
	path = out_dir / filename
	if path.exists() and not overwrite:
		print(f"[load] {path}")
		with path.open("rb") as f:
			return pickle.load(f)

	print(f"[train] {path} not found – creating new agent.")
	obj = train_fn(*build_args, **build_kwargs)

	with path.open("wb") as f:
		pickle.dump(obj, f)
	print(f"[save] {path}")

	return obj

def train_agent(env, epochs, random_action_rate_decay):
    agent = BlackjackAgent(env, random_action_rate_decay=random_action_rate_decay)
    for epoch in tqdm(range(epochs)):
        play_one_hand(env, agent, mode=Mode.TRAIN, seed=epoch)
        agent.decay_rar()

    if not env.hitorstand_only:
        for state, vals in agent.q_table.items():
            if state[2] == 0 and state[0] % 2 != 0:
                agent.q_table[state][ACTIONS.SPLIT] = -np.inf
            elif state[2] == 1 and state[0] != 12:
                agent.q_table[state][ACTIONS.SPLIT] = -np.inf
            
    return agent

def plot_strategy(q_table, title=""):
	player_vals = np.array([key[0] for key in q_table.keys()])
	dealer_vals = np.array([key[1] for key in q_table.keys()])
	action_vals = np.array([max(value) for value in q_table.values()])
	actions = np.array([np.argmax(value) for value in q_table.values()])

	mask = player_vals <= 21
	player_vals = player_vals[mask]
	dealer_vals = dealer_vals[mask]
	action_vals = action_vals[mask]
	actions = actions[mask]

	fig = plt.figure(figsize=(20, 10))    

	ax1 = fig.add_subplot(121, projection='3d')
	surf = ax1.plot_trisurf(dealer_vals, player_vals, action_vals, cmap='viridis', edgecolor='none')

	ax1.set_ylim(min(player_vals), 21)
	ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
	ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

	x_ticks = np.arange(1, 11, 1)
	ax1.set_xticks(x_ticks)
	x_labels = [str(x) for x in x_ticks]
	x_labels[0] = 'A'
	ax1.set_xticklabels(x_labels)

	# Add labels and title
	ax1.set_xlabel('Dealer Upcard')
	ax1.set_ylabel('Player Sum')
	ax1.set_zlabel('Optimal Action value')

	df = pd.DataFrame({
		'Player Sum': player_vals,   
		'Dealer Upcard': dealer_vals,  
		'actions': actions,
	})

	df = df[df['Player Sum'] <= 21]

	heatmap_data = df.pivot(index='Dealer Upcard', columns='Player Sum', values='actions')

	ax2 = fig.add_subplot(122)
	n_actions = max(actions) + 1
	cmap = mpl.colors.ListedColormap(sns.color_palette("viridis", n_actions))
	boundaries = np.arange(-0.5, n_actions + 0.5, 1)
	norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
	cax = fig.add_axes([1.01, 0.25, 0.025, 0.5])

	hm = sns.heatmap(
			heatmap_data,
			annot=True,
			cmap=cmap,
			norm=norm,
			ax=ax2,
			cbar=True,
			cbar_ax=cax)
    
    # label bar ticks
	cbar = hm.collections[0].colorbar
	cbar.set_ticks(range(n_actions))
	if n_actions == 5:
		ticklabels = ["0 Stand", "1 Hit", "2 Double", "3 Surrender", "4 Split"]
	elif n_actions == 2:
		ticklabels = ["0 Stand", "1 Hit"]

	cbar.set_ticklabels(ticklabels)

	fig.suptitle(title, fontsize=20, ha='center')
	fig.tight_layout() 
	fig.colorbar(surf, shrink=0.3, aspect=5)

	code_dir = Path(__file__).resolve().parent
	image_dir = code_dir.parent / "image"
	image_dir.mkdir(exist_ok=True)
	outfile = image_dir / f"{title}.png"
	fig.savefig(outfile, dpi=300, bbox_inches="tight")
	print(f"[saved] {outfile}")

	# plt.show()
    

def summarize_reward_sets(rewards_all_actions,
                        rewards_hit_stand,
                        rewards_basic,
                        strategy_names=("Basic", "Hit / Stand", "All Actions")
                        ):
    def _stats(arr):
        arr        = np.asarray(arr, dtype=float)
        n          = arr.size
        mean       = arr.mean()
        std        = np.std(arr)

        wins       = (arr > 0).sum()
        ties       = (arr == 0).sum()
        losses     = n - wins - ties

        win_rate   = wins   / n
        tie_rate   = ties   / n
        loss_rate  = losses / n

        return [n, mean, std, win_rate, tie_rate, loss_rate]

    data = [
        _stats(rewards_basic),
        _stats(rewards_hit_stand),
        _stats(rewards_all_actions),
    ]

    cols = ["Hands played",
            "Mean reward",
            "Std. deviation",
            "Win rate",
            "Tie rate",
            "Loss rate"]

    return pd.DataFrame(data, index=strategy_names, columns=cols)

def plot_agent_performance(outcomes, bins=15, title=""):
	mu   = outcomes.mean()  
	fig = plt.figure(figsize=(8, 5))
	sns.histplot(outcomes, bins=bins, stat="density", kde=True, alpha=0.6, edgecolor="k")
	plt.axvline(mu, color="red", linestyle="--", linewidth=1.5, label=f"mean = {mu:.3f}")
	plt.title(title)
	plt.xlabel("Reward per hand")
	plt.ylabel("Density")
	plt.legend()
	plt.tight_layout()

	code_dir = Path(__file__).resolve().parent
	image_dir = code_dir.parent / "image"
	image_dir.mkdir(exist_ok=True)
	outfile = image_dir / f"{title}.png"
	fig.savefig(outfile, dpi=300, bbox_inches="tight")
	print(f"[saved] {outfile}")