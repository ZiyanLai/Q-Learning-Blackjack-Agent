import numpy as np
from tqdm import tqdm

from blackjackenv   import MyBlackJackEnv
from blackjackagent import BlackjackAgent, BlackjackBasicAgent
from utils          import *

EPOCHS = 50000000
HALF_LIFE = 0.10
RANDOM_DECAY = np.exp(np.log(0.5) / (EPOCHS * HALF_LIFE))
NUM_GAMES = 500_000

def main():
    # ---------- train or load agents ----------
    env_all = MyBlackJackEnv(sab=True)
    agent_all = get_or_train("agent_allactions.pkl",
                            train_agent,
                            env_all, EPOCHS, RANDOM_DECAY,
                            overwrite=False)

    env_hs  = MyBlackJackEnv(sab=True, hitorstand_only=True)
    agent_hs = get_or_train("agent_hitorstand.pkl",
                            train_agent,
                            env_hs, EPOCHS, RANDOM_DECAY,
                            overwrite=False)

    # ---------- plot strategies ----------
    for agent, name in [(agent_hs,  "Hit-or-Stand Agent"),
                        (agent_all, "All-Actions Agent")]:
        soft  = {k: v for k, v in agent.q_table.items() if k[2] == 1}
        hard  = {k: v for k, v in agent.q_table.items() if k[2] == 0}
        plot_strategy(soft, title=f"{name}: Soft Hand")
        plot_strategy(hard, title=f"{name}: Hard Hand")

    # ---------- evaluate performance ----------
    rewards_all = np.array([play_one_hand(env_all, agent_all,  mode=Mode.TEST, seed=i+1)
                            for i in tqdm(range(NUM_GAMES))])
    rewards_hs  = np.array([play_one_hand(env_hs,  agent_hs,   mode=Mode.TEST, seed=i+1)
                            for i in tqdm(range(NUM_GAMES))])
    agent_basic = BlackjackBasicAgent()
    rewards_bas = np.array([play_one_hand(env_all, agent_basic, mode=Mode.TEST, seed=i+1)
                            for i in tqdm(range(NUM_GAMES))])

    summarize_reward_sets(rewards_all, rewards_hs, rewards_bas)
    plot_agent_performance(rewards_all, title="All-Actions Agent Reward Distribution")
    plot_agent_performance(rewards_hs, title="Hit-or-Stand Agent Reward Distribution")
    plot_agent_performance(rewards_bas, title="Basic Agent Reward Distribution")

# ----- standard entry point -----
if __name__ == "__main__":
    main()