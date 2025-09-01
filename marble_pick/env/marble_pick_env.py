from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np
from collections import deque
from enum import Enum


class DIRECTION(Enum):
    LEFT = 0
    RIGHT = 1


class MarblePickEnv(ParallelEnv):
    metadata = {
        "name": "marble_pick_v0",
    }

    def __init__(self, num_agents=8, num_bags=3, marbles_per_bag=15):
        # Define the number of agents
        self.number_of_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self.number_of_agents)]
        self.agents = self.possible_agents

        # Define the game's constants
        self.num_bags = num_bags
        self.marbles_per_bag = marbles_per_bag
        self.num_colors = 5
        self.num_properties = self.num_colors + 2  # Color (one-hot), Size, Weight

        # Define the observation and action spaces for each agent
        # The action space is a simple discrete choice from the current bag
        self.action_spaces = {
            agent: Discrete(self.marbles_per_bag) for agent in self.possible_agents
        }

        single_marble_low = np.array(
            [0 for _ in range(self.num_colors)] + [1, 1], dtype=np.float32
        )
        single_marble_high = np.array(
            [1 for _ in range(self.num_colors)] + [10, 7], dtype=np.float32
        )

        # The observation space is a dictionary
        self.observation_spaces = {
            agent: Dict(
                {
                    "current_bag": Box(
                        low=np.tile(single_marble_low, (self.marbles_per_bag, 1)),
                        high=np.tile(single_marble_high, (self.marbles_per_bag, 1)),
                        shape=(self.marbles_per_bag, self.num_properties),
                        dtype=np.float32,
                    ),
                    "my_collection": Box(
                        low=np.tile(
                            single_marble_low,
                            (self.marbles_per_bag * self.num_bags, 1),
                        ),
                        high=np.tile(
                            single_marble_high,
                            (self.marbles_per_bag * self.num_bags, 1),
                        ),
                        shape=(
                            self.marbles_per_bag * self.num_bags,
                            self.num_properties,
                        ),
                        dtype=np.float32,
                    ),
                    "action_mask": Box(
                        low=0,
                        high=1,
                        shape=(self.marbles_per_bag,),
                        dtype=np.int8,
                    ),
                }
            )
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        # generate all the marbles for the entire game
        all_marbles = self._generate_marbles()

        # shuffle and distribute the marbles into bags for each agent
        self.agent_bags = {}
        np.random.shuffle(all_marbles)
        for i in range(self.num_agents):
            # find the start and end indices for the current agent's marbles
            # start = agent's idx * marbles per bag * num bags, ex: (1) * (15) * (3) = 45, (0) * (15) * (3) = 0
            start_idx = i * self.marbles_per_bag * self.num_bags
            # end = start + marbles per bag * num bags, ex: (45) + (15) * (3) = 90, (0) + (15) * (3) = 45
            end_idx = start_idx + self.marbles_per_bag * self.num_bags
            # reshape the marbles into bags for the current agent 45 x 3 --> 3 x 15 x 7 (the last dim is the properties --> [color (one hot), size, weight])
            agent_marbles = all_marbles[start_idx:end_idx].reshape(
                self.num_bags, self.marbles_per_bag, self.num_properties
            )
            self.agent_bags[self.possible_agents[i]] = agent_marbles

        # init the agent collections and other state vars
        self.agent_collections = {agent: [] for agent in self.possible_agents}
        self.current_bag_idx = 0
        self.agents = self.possible_agents[:]

        # return the initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        info = {agent: {} for agent in self.agents}
        return observations, info

    def _generate_marbles(self):
        # placeholder to generate marbles
        total_marbles = self.number_of_agents * self.marbles_per_bag * self.num_bags
        # create an array of one-hot encoded colors
        one_hots = np.eye(self.num_colors)
        # generate all marble colors
        marble_color_indices = np.random.choice(
            self.num_colors, size=total_marbles, replace=True
        )
        marble_colors = one_hots[marble_color_indices]
        # generate all marble sizes
        marble_sizes = np.random.randint(1, 11, size=total_marbles).reshape(-1, 1)
        # generate all marble weights
        marble_weights = np.random.randint(1, 6, size=total_marbles).reshape(-1, 1)
        # horizontally stack the arrays
        marbles = np.hstack((marble_colors, marble_sizes, marble_weights))
        print("MARBLES", marbles)
        return marbles

    def _get_observation(self, agent):
        # helper fn to create the observation for a given agent
        current_bag = self.agent_bags[agent][self.current_bag_idx]
        my_collection = (
            np.array(self.agent_collections[agent])
            if self.agent_collections[agent]
            else np.zeros((0, self.num_properties))
        )

        # pad my_collection to the maximum size
        padded_collection = np.zeros(
            (self.marbles_per_bag * self.num_bags, self.num_properties)
        )
        padded_collection[: my_collection.shape[0]] = my_collection

        # create the action mask given the current_bag
        action_mask = np.ones(current_bag.shape[0], dtype=bool)
        for i in range(current_bag.shape[0]):
            if np.all(current_bag[i] == 0):
                action_mask[i] = False

        return {
            "current_bag": current_bag,
            "my_collection": padded_collection,
            "action_mask": action_mask,
        }

    def _get_reward(self, agent) -> np.float32:
        """Returns the reward for the agent.

        This is 0 until the end of the game. (naive approach)

        Args:
            agent (str): The agent for which to calculate the reward.

        Returns:
            float: The reward for the agent.
        """
        # find the two most populous colors in the agent's collection
        collection = self.agent_collections[agent]
        # collection is shaped like (num_marbles x num_properties)
        colors = {
            0: [1, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0],
            2: [0, 0, 1, 0, 0],
            3: [0, 0, 0, 1, 0],
            4: [0, 0, 0, 0, 1],
        }

        # get the marble counts by color
        marble_counts = np.sum(np.array(collection)[:, :5], axis=0)

        # find the two most populous colors
        most_populous_indices = np.argsort(marble_counts)[-2:]

        # create boolean masks for the two colors
        mask_one = np.all(
            np.array(collection)[:, :5] == colors[most_populous_indices[0]], axis=1
        )
        mask_two = np.all(
            np.array(collection)[:, :5] == colors[most_populous_indices[1]], axis=1
        )
        combined_mask = np.logical_or(mask_one, mask_two)

        # filter the collection based on the combined mask
        filtered_collection = np.array(collection)[combined_mask]

        # get the sum over rows
        sum_over_rows = np.sum(filtered_collection, axis=0)

        total_size = sum_over_rows[-2]
        total_weight = sum_over_rows[-1]

        positive_reward = total_size
        curve_over_3 = (total_weight / len(filtered_collection)) - 3
        negative_reward = curve_over_3 * 10 if curve_over_3 > 0 else 0

        return np.float32(positive_reward - negative_reward)

    def step(self, actions):
        """Performs a single timestep of the environment

        Args:
            actions (dict): A dictionary mapping agents to their actions.
        """
        # step 1: process each of the actions and update the agent's collection
        for agent, action in actions.items():
            marble_picked = self.agent_bags[agent][self.current_bag_idx][action]
            self.agent_collections[agent].append(np.copy(marble_picked))
            self.agent_bags[agent][self.current_bag_idx][action] = np.zeros(
                self.num_properties
            )

        # step 2: update the environment (passing the bags)
        # self.agent_bags looks like (num_agents (keys) x num_bags x num_marbles_per_bag x num_properties)
        direction: DIRECTION = (
            DIRECTION.LEFT if self.current_bag_idx % 2 == 0 else DIRECTION.RIGHT
        )
        agent_bags_list = deque(
            [
                self.agent_bags[agent][self.current_bag_idx]
                for agent in self.possible_agents
            ]
        )
        if direction == DIRECTION.LEFT:
            agent_bags_list.rotate(-1)
        else:
            agent_bags_list.rotate(1)

        for i, agent in enumerate(self.possible_agents):
            self.agent_bags[agent][self.current_bag_idx] = agent_bags_list[i]

        # check if all bags are empty
        terminated = False
        if all(np.all(bag == 0) for bag in agent_bags_list):
            self.current_bag_idx += 1
            if self.current_bag_idx == self.num_bags:
                terminated = True

        if terminated:
            observations = {a: None for a in self.agents}
        else:
            observations = {a: self._get_observation(a) for a in self.agents}
        rewards = {
            a: (0 if not terminated else self._get_reward(a)) for a in self.agents
        }
        terminations = {a: terminated for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        print("\n--- Current Game State ---")
        print(f"Current Bag Round: {self.current_bag_idx + 1}")
        for agent in self.possible_agents:
            collection = self.agent_collections[agent]
            if self.current_bag_idx == self.num_bags:
                bag = np.array([])
            else:
                bag = self.agent_bags[agent][self.current_bag_idx]
            print(f"\n{agent}:")
            print(f"  Marbles Collected: {len(collection)}")
            if len(bag):
                # You can add more detailed info about the collection if you want
                print(
                    f"  Current Bag Contents: {len(bag[np.any(bag > 0, axis=1)])} marbles remain"
                )
            else:
                reward = self._get_reward(agent)
                print(f"  Reward: {reward}")
            # Print a simple visual representation of the bag
        print("--------------------------")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
