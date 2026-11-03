import importlib
import sys
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

# Attempt to import the agent implementation. Adjust the import path if the
# project structure differs.
try:
    TrafficLightAgent = importlib.import_module("agent").TrafficLightAgent
except Exception:
    try:
        TrafficLightAgent = importlib.import_module("agents.traffic_light_agent").TrafficLightAgent
    except Exception as e:
        pytest.fail(f"Could not import TrafficLightAgent: {e}")


def make_dummy_env():
    """
    Create a minimal Gymnasium environment used for testing.
    Observation space: Box(0, 1, shape=(4,))
    Action space: Discrete(2)
    """
    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            obs = self.observation_space.sample()
            return obs, {}

        def step(self, action):
            # Return a random next observation, reward, done flag, and info.
            next_obs = self.observation_space.sample()
            reward = float(np.random.rand())
            terminated = np.random.rand() < 0.05
            truncated = False
            info = {}
            return next_obs, reward, terminated, truncated, info

    return DummyEnv()


def get_policy_parameters(agent):
    """
    Retrieve a flat list of the agent's policy network parameters.
    The function tries common attribute names used in RL agents.
    """
    # Common attribute names for the policy network
    for attr_name in ("policy", "policy_network", "model", "actor_critic"):
        if hasattr(agent, attr_name):
            policy = getattr(agent, attr_name)
            if isinstance(policy, torch.nn.Module):
                return list(policy.parameters())
            # Stable‑Baselines3 agents expose a ``policy`` attribute that
            # itself has a ``state_dict`` method.
            if hasattr(policy, "state_dict"):
                # Convert state dict tensors to parameters
                return [torch.nn.Parameter(v) for v in policy.state_dict().values() if isinstance(v, torch.Tensor)]
    # Fallback: try to locate any torch.nn.Module attribute
    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            return list(value.parameters())
    raise AttributeError("Unable to locate policy network parameters in the agent.")


def test_action_selection():
    """
    Verify that the agent returns a valid action for a given observation.
    The action must be an integer within the environment's action space.
    """
    env = make_dummy_env()
    obs, _ = env.reset(seed=42)

    # Instantiate the agent. Most agents accept observation and action spaces.
    agent = TrafficLightAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=1e-3,
        gamma=0.99,
        seed=42,
    )

    # The agent may expose different method names for action selection.
    if hasattr(agent, "select_action"):
        action = agent.select_action(obs)
    elif hasattr(agent, "act"):
        action = agent.act(obs)
    else:
        pytest.fail("Agent does not provide a method for action selection (select_action or act).")

    # Ensure the action is a scalar integer and lies inside the action space.
    assert isinstance(action, (int, np.integer)), f"Action type is {type(action)}, expected int."
    assert env.action_space.contains(action), f"Action {action} is not within the action space {env.action_space}."


def test_learning_update():
    """
    Ensure that calling the agent's learning routine updates the policy network.
    The test creates a tiny replay buffer, performs one learning step, and checks
    that at least one parameter tensor has changed.
    """
    env = make_dummy_env()
    obs, _ = env.reset(seed=123)

    agent = TrafficLightAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=1e-3,
        gamma=0.99,
        seed=123,
    )

    # Record initial parameters.
    initial_params = [p.clone().detach() for p in get_policy_parameters(agent)]

    # Build a minimal batch of transitions.
    batch_size = 5
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for _ in range(batch_size):
        a = (
            agent.select_action(obs)
            if hasattr(agent, "select_action")
            else agent.act(obs)
        )
        next_obs, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        observations.append(obs)
        actions.append(a)
        rewards.append(reward)
        next_observations.append(next_obs)
        dones.append(done)

        obs = next_obs if not done else env.reset(seed=123)[0]

    # Convert batch to tensors (common format for most agents).
    obs_tensor = torch.tensor(observations, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    next_obs_tensor = torch.tensor(next_observations, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

    # Perform a learning step. Different agents expose different APIs.
    if hasattr(agent, "learn"):
        # Assume the method accepts a dictionary similar to Stable‑Baselines3.
        try:
            agent.learn(
                obs=obs_tensor,
                actions=actions_tensor,
                rewards=rewards_tensor,
                next_obs=next_obs_tensor,
                dones=dones_tensor,
            )
        except TypeError:
            # Fallback to a generic ``learn`` that takes a single batch tuple.
            agent.learn((obs_tensor, actions_tensor, rewards_tensor, next_obs_tensor, dones_tensor))
    elif hasattr(agent, "update"):
        agent.update(
            observations=obs_tensor,
            actions=actions_tensor,
            rewards=rewards_tensor,
            next_observations=next_obs_tensor,
            dones=dones_tensor,
        )
    else:
        pytest.fail("Agent does not provide a learning method (learn or update).")

    # Record parameters after learning.
    updated_params = get_policy_parameters(agent)

    # At least one parameter should have changed.
    changes = [
        not torch.allclose(p0, p1, atol=1e-8, rtol=1e-5)
        for p0, p1 in zip(initial_params, updated_params)
    ]
    assert any(changes), "No policy parameters changed after learning step."
