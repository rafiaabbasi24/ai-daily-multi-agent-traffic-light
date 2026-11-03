import importlib
import itertools
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pytest


def _load_traffic_env_class() -> Any:
    """
    Attempt to import the Traffic environment class from common locations.
    The project may expose the environment under different module names;
    this helper tries a few plausible import paths and raises an informative
    error if none succeed.
    """
    possible_paths = [
        "traffic_env.TrafficEnv",
        "envs.traffic_env.TrafficEnv",
        "dynamic_traffic.env.TrafficEnv",
        "dynamic_multi_agent_traffic_control.env.TrafficEnv",
    ]

    for path in possible_paths:
        module_name, class_name = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            env_cls = getattr(module, class_name)
            return env_cls
        except (ImportError, AttributeError):
            continue

    raise ImportError(
        "Could not locate the TrafficEnv class. Tried the following import paths: "
        + ", ".join(possible_paths)
    )


@pytest.fixture(scope="module")
def TrafficEnv():
    """Fixture that provides the TrafficEnv class."""
    return _load_traffic_env_class()


@pytest.fixture
def env(TrafficEnv):
    """
    Fixture that creates a fresh environment instance for each test.
    The environment is created with a deterministic seed to make the tests
    reproducible.
    """
    env = TrafficEnv(seed=42)
    yield env
    env.close()


def _sample_action(action_space: gym.spaces.Space) -> Any:
    """Utility to sample a valid action from any Gymnasium action space."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.sample()
    if isinstance(action_space, gym.spaces.Box):
        return np.clip(action_space.sample(), action_space.low, action_space.high)
    if isinstance(action_space, gym.spaces.MultiBinary):
        return action_space.sample()
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return action_space.sample()
    if isinstance(action_space, gym.spaces.Tuple):
        return tuple(_sample_action(s) for s in action_space.spaces)
    if isinstance(action_space, gym.spaces.Dict):
        return {k: _sample_action(v) for k, v in action_space.spaces.items()}
    raise TypeError(f"Unsupported action space type: {type(action_space)}")


def _assert_observation_match_space(obs: Any, space: gym.spaces.Space):
    """Recursively assert that an observation conforms to a given Gymnasium space."""
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(obs, dict), "Observation should be a dict for Dict space"
        assert set(obs.keys()) == set(space.spaces.keys()), "Observation keys mismatch"
        for k, subspace in space.spaces.items():
            _assert_observation_match_space(obs[k], subspace)
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs, tuple), "Observation should be a tuple for Tuple space"
        assert len(obs) == len(space.spaces), "Observation length mismatch"
        for sub_obs, subspace in zip(obs, space.spaces):
            _assert_observation_match_space(sub_obs, subspace)
    elif isinstance(space, gym.spaces.Box):
        assert isinstance(obs, np.ndarray), "Box observation must be a NumPy array"
        assert obs.shape == space.shape, f"Box observation shape mismatch: {obs.shape} vs {space.shape}"
        assert obs.dtype == space.dtype, f"Box observation dtype mismatch: {obs.dtype} vs {space.dtype}"
        # Values should be within bounds (allow small numerical tolerance)
        assert np.all(obs >= space.low - 1e-6) and np.all(obs <= space.high + 1e-6), "Box observation out of bounds"
    elif isinstance(space, gym.spaces.Discrete):
        assert isinstance(obs, (int, np.integer)), "Discrete observation must be an integer"
        assert 0 <= int(obs) < space.n, "Discrete observation out of range"
    elif isinstance(space, gym.spaces.MultiBinary):
        assert isinstance(obs, np.ndarray), "MultiBinary observation must be a NumPy array"
        assert obs.shape == space.shape, "MultiBinary observation shape mismatch"
        assert np.all(np.logical_or(obs == 0, obs == 1)), "MultiBinary observation contains values other than 0/1"
    elif isinstance(space, gym.spaces.MultiDiscrete):
        assert isinstance(obs, np.ndarray), "MultiDiscrete observation must be a NumPy array"
        assert obs.shape == space.shape, "MultiDiscrete observation shape mismatch"
        for val, n in zip(obs, space.nvec):
            assert 0 <= val < n, "MultiDiscrete observation value out of range"
    else:
        raise TypeError(f"Unsupported observation space type: {type(space)}")


def test_environment_initialization(env):
    """Check that the environment instance exposes the required Gymnasium attributes."""
    assert isinstance(env, gym.Env), "Environment should inherit from gym.Env"
    assert isinstance(env.observation_space, gym.spaces.Space), "Missing or invalid observation_space"
    assert isinstance(env.action_space, gym.spaces.Space), "Missing or invalid action_space"


def test_reset_returns_valid_observation(env):
    """After reset, the observation must conform to the declared observation space."""
    obs, info = env.reset()
    # Gymnasium's reset returns (obs, info) tuple
    assert info is not None, "Reset should return an info dict (may be empty)"
    _assert_observation_match_space(obs, env.observation_space)


def test_step_returns_valid_tuple(env):
    """A single step must return (observation, reward, terminated, truncated, info)."""
    obs, _ = env.reset()
    action = _sample_action(env.action_space)
    step_result = env.step(action)

    # Gymnasium step returns a 5‑tuple
    assert isinstance(step_result, tuple) and len(step_result) == 5, "step must return (obs, reward, terminated, truncated, info)"
    next_obs, reward, terminated, truncated, info = step_result

    # Validate each component
    _assert_observation_match_space(next_obs, env.observation_space)
    assert isinstance(reward, (float, np.floating, int, np.integer)), "Reward must be a numeric scalar"
    assert np.isfinite(reward), "Reward must be a finite number"
    assert isinstance(terminated, bool), "terminated flag must be bool"
    assert isinstance(truncated, bool), "truncated flag must be bool"
    assert isinstance(info, dict), "info must be a dict"


def test_episode_termination(env):
    """
    Run an episode until termination or truncation.
    The environment should eventually signal the end of an episode
    (either terminated or truncated) within a reasonable horizon.
    """
    obs, _ = env.reset()
    max_steps = 5000  # generous upper bound to avoid infinite loops
    for step_idx in range(max_steps):
        action = _sample_action(env.action_space)
        obs, reward, terminated, truncated, info = env.step(action)

        # Ensure step outputs stay valid throughout the episode
        _assert_observation_match_space(obs, env.observation_space)
        assert isinstance(reward, (float, np.floating, int, np.integer))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        if terminated or truncated:
            # Verify that after a done flag the environment can be reset without error
            env.reset()
            break
    else:
        pytest.fail(f"Environment did not terminate after {max_steps} steps")


def test_seed_reproducibility(TrafficEnv):
    """
    Two environments seeded identically should produce identical trajectories
    when given the same sequence of actions.
    """
    seed = 12345
    env_a = TrafficEnv(seed=seed)
    env_b = TrafficEnv(seed=seed)

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    _assert_observation_match_space(obs_a, env_a.observation_space)
    _assert_observation_match_space(obs_b, env_b.observation_space)

    # Observations after reset should be identical (or at least statistically indistinguishable)
    np.testing.assert_allclose(obs_a, obs_b, err_msg="Observations differ after identical seeding")

    # Generate a deterministic sequence of actions
    rng = np.random.default_rng(seed)
    actions = [_sample_action(env_a.action_space) for _ in range(20)]

    for a in actions:
        next_a, rew_a, term_a, trunc_a, _ = env_a.step(a)
        next_b, rew_b, term_b, trunc_b, _ = env_b.step(a)

        _assert_observation_match_space(next_a, env_a.observation_space)
        _assert_observation_match_space(next_b, env_b.observation_space)

        np.testing.assert_allclose(next_a, next_b, err_msg="Next observations diverge")
        assert np.isclose(rew_a, rew_b), "Rewards diverge"
        assert term_a == term_b, "Termination flags diverge"
        assert trunc_a == trunc_b, "Truncation flags diverge"

    env_a.close()
    env_b.close()


def test_invalid_action_raises(env):
    """
    Providing an action outside the defined action space should raise an error.
    The exact exception type may vary (ValueError, AssertionError, etc.),
    but it must be a subclass of Exception.
    """
    space = env.action_space

    # Construct an out‑of‑bounds action depending on the space type
    if isinstance(space, gym.spaces.Discrete):
        invalid_action = space.n  # one past the highest valid index
    elif isinstance(space, gym.spaces.Box):
        # Shift the sampled action beyond the upper bound
        sample = space.sample()
        invalid_action = sample + (space.high - space.low) * 2.0
        invalid_action = np.clip(invalid_action, space.low - 10, space.high + 10)
    elif isinstance(space, gym.spaces.MultiBinary):
        invalid_action = np.full(space.shape, 2, dtype=space.dtype)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        invalid_action = np.array([n for n in space.nvec])  # one past each max
    elif isinstance(space, gym.spaces.Tuple):
        invalid_action = tuple(_sample_action(s) for s in space.spaces)
        # corrupt the first sub‑action
        if isinstance(space.spaces[0], gym.spaces.Discrete):
            invalid_action = list(invalid_action)
            invalid_action[0] = space.spaces[0].n
            invalid_action = tuple(invalid_action)
    elif isinstance(space, gym.spaces.Dict):
        invalid_action = {k: _sample_action(v) for k, v in space.spaces.items()}
        # corrupt one entry
        first_key = next(iter(space.spaces))
        subspace = space.spaces[first_key]
        if isinstance(subspace, gym.spaces.Discrete):
            invalid_action[first_key] = subspace.n
        elif isinstance(subspace, gym.spaces.Box):
            invalid_action[first_key] = subspace.high + 1.0
    else:
        pytest.skip(f"Unsupported action space type for invalid‑action test: {type(space)}")

    with pytest.raises(Exception):
        env.step(invalid_action)