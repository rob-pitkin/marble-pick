import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


from marble_pick.env.marble_pick_env import MarblePickEnv
import numpy as np


def main():
    env = MarblePickEnv()
    observations, infos = env.reset()
    done = {a: False for a in infos}
    agents = [a for a in env.agents]
    while not any(done.values()):
        actions = {a: -1 for a in agents}
        for a in agents:
            # get an action mask
            action_mask = np.array(observations[a]["action_mask"], dtype=np.int8)
            action = env.action_space(a).sample(action_mask)
            actions[a] = action
        observations, rewards, done, truncations, infos = env.step(actions)
        env.render()
    env.close()


if __name__ == "__main__":
    main()
