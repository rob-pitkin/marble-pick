# MarblePick

`MarblePick` is a multi-agent reinforcement learning environment implemented using the `pettingzoo` library. It models a scenario where multiple agents strategically pick marbles from shared, rotating bags to maximize their final collection's score.

This environment is designed to explore cooperative and competitive dynamics in a resource-gathering context with delayed rewards and complex state representations.

## Installation

This project uses `uv` for environment and dependency management.

## Usage

You can run the environment with random agents using the provided `test.py` script

## Environment Details

-   **Agents**: 8 (`num_agents`)
-   **Bags per Agent**: 3 (`num_bags`)
-   **Marbles per Bag**: 15 (`marbles_per_bag`)

The game proceeds in rounds. In each step, every agent picks one marble from their currently held bag. After the picks, the bags are rotated among the agents. A round ends when a bag is empty, and the next bag is brought into play. The game terminates when all marbles from all bags have been picked.

### Observation Space

The observation for each agent is a `Dict` space with three components:

-   `current_bag`: A `Box` of shape `(15, 7)` representing the marbles in the bag currently held by the agent. Each row is a marble with 7 properties.
-   `my_collection`: A `Box` of shape `(45, 7)` representing the marbles the agent has collected so far. It is padded with zeros for marbles not yet collected.
-   `action_mask`: A `Box` of shape `(15,)` with a binary mask indicating which marbles in the `current_bag` are available to be picked (1 for available, 0 for already picked).

Each marble is a 7-element vector:
-   **Color**: A 5-element one-hot encoded vector.
-   **Size**: A float value between 1 and 10.
-   **Weight**: A float value between 1 and 7.

### Action Space

The action space for each agent is `Discrete(15)`, where the integer corresponds to the index of the marble to pick from the `current_bag`. The `action_mask` in the observation should be used to prevent picking invalid (already taken) marbles.

### Reward Function

Rewards are only distributed at the end of the game when all marbles have been collected. The final reward for each agent is calculated as follows:

1.  **Positive Reward**: The sum of the `size` values for all marbles belonging to the **two most common colors** in the agent's final collection.
2.  **Negative Reward**: A penalty based on the average `weight` of the agent's final collection (**two most common colors**). The penalty is calculated as `10 * max(0, avg_weight - 3)`.

The total reward is `positive_reward - negative_reward`.
