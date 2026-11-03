#!/usr/bin/env python3
"""
src/main.py
Entry point for the Dynamic Multi‑Agent Traffic Light Control demo.

This script can be used to:
* Train a reinforcement‑learning agent (PPO) on the traffic‑light environment.
* Visualise a trained policy by rendering the environment.
* Plot training statistics (episode reward over time).

The implementation relies only on publicly available packages:
    - gymnasium
    - torch
    - stable‑baselines3
    - numpy
    - matplotlib
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any, Callable, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


def _import_traffic_env() -> Callable[[], gym.Env]:
    """
    Import the custom traffic‑light environment.

    The project is expected to expose a class named ``TrafficLightEnv`` in a
    module ``src.envs.traffic_env``.  If the import fails, a very small fallback
    environment is returned so that the demo script can still run without the
    full simulation stack.

    Returns
    -------
    Callable[[], gym.Env]
        A zero‑argument factory that creates a fresh environment instance.
    """
    try:
        from src.envs.traffic_env import TrafficLightEnv  # type: ignore

        def factory() -> gym.Env:
            return TrafficLightEnv()

    except Exception as exc:  # pragma: no cover
        # Fallback minimal environment – useful for CI or when the full env is not
        # installed.  The environment mimics a single traffic light with two
        # actions: stay red (0) or turn green (1).  Rewards are random to keep the
        # interface functional.
        class DummyTrafficEnv(gym.Env):
            """A minimal stand‑in for the real traffic‑light environment."""

            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                # Observation: a single scalar representing traffic density (0‑1)
                self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                # Action: 0 = keep red, 1 = turn green
                self.action_space = gym.spaces.Discrete(2)
                self._step = 0
                self._max_steps = 50

            def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
                super().reset(seed=seed)
                self._step = 0
                obs = self.observation_space.sample()
                return obs, {}

            def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
                self._step += 1
                # Random traffic density for next observation
                obs = self.observation_space.sample()
                # Simple reward: +1 for green when traffic is high, -1 otherwise
                traffic = obs[0]
                reward = (1.0 if action == 1 else -1.0) * (traffic - 0.5)
                terminated = self._step >= self._max_steps
                truncated = False
                info: dict[str, Any] = {}
                return obs, float(reward), terminated, truncated, info

            def render(self) -> None:  # pragma: no cover
                pass

            def close(self) -> None:  # pragma: no cover
                pass

        def factory() -> gym.Env:
            return DummyTrafficEnv()

        print(f"[WARN] Could not import the real traffic environment: {exc}", file=sys.stderr)
        print("[WARN] Falling back to a dummy environment for demonstration purposes.", file=sys.stderr)

    return factory


def make_vec_env(env_factory: Callable[[], gym.Env], n_envs: int = 1) -> VecEnv:
    """
    Wrap a gymnasium environment factory into a vectorised environment.

    Parameters
    ----------
    env_factory : Callable[[], gym.Env]
        Function that creates a fresh environment instance.
    n_envs : int, optional
        Number of parallel environments (default is 1).

    Returns
    -------
    VecEnv
        A vectorised environment compatible with Stable‑Baselines3.
    """
    return DummyVecEnv([env_factory for _ in range(n_envs)])


class RewardLoggerCallback(BaseCallback):
    """
    Callback that records the total reward per episode and stores it in
    ``self.rewards`` for later visualisation.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.rewards: list[float] = []
        self._episode_reward: float = 0.0

    def _on_step(self) -> bool:
        # ``self.locals["rewards"]`` is a numpy array with shape (n_envs,)
        reward = self.locals["rewards"]
        self._episode_reward += float(np.mean(reward))
        return True

    def _on_rollout_end(self) -> None:
        # Called after each rollout; we treat it as an episode boundary for
        # simplicity because the dummy env terminates after a fixed number of steps.
        self.rewards.append(self._episode_reward)
        self._episode_reward = 0.0


def plot_rewards(rewards: list[float], save_path: pathlib.Path | None = None) -> None:
    """
    Plot episode rewards over time.

    Parameters
    ----------
    rewards : list[float]
        Collected episode rewards.
    save_path : pathlib.Path | None, optional
        If provided, the figure is saved to this location.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker="o", linestyle="-")
    plt.title("Training Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Reward plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# --------------------------------------------------------------------------- #
# Core functionality
# --------------------------------------------------------------------------- #


def train(
    env_factory: Callable[[], gym.Env],
    total_timesteps: int,
    model_path: pathlib.Path,
    log_dir: pathlib.Path,
    n_envs: int = 1,
) -> PPO:
    """
    Train a PPO agent on the provided environment.

    Parameters
    ----------
    env_factory : Callable[[], gym.Env]
        Factory that returns a fresh environment instance.
    total_timesteps : int
        Number of environment steps to train for.
    model_path : pathlib.Path
        Destination file for the trained model (``.zip`` format).
    log_dir : pathlib.Path
        Directory where Stable‑Baselines3 logs will be written.
    n_envs : int, optional
        Number of parallel environments (default 1).

    Returns
    -------
    PPO
        The trained model.
    """
    # Ensure directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure the SB3 logger to write CSV files for easy post‑processing
    new_logger = configure(str(log_dir), ["stdout", "csv"])

    vec_env = make_vec_env(env_factory, n_envs=n_envs)

    # Callback chain: checkpoint every 100k steps + reward logging
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_traffic",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    reward_logger_cb = RewardLoggerCallback()
    callbacks = CallbackList([checkpoint_cb, reward_logger_cb])

    # PPO hyper‑parameters are deliberately modest to keep training fast.
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.set_logger(new_logger)

    print("[INFO] Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    print("[INFO] Training finished. Saving model...")
    model.save(str(model_path))

    # Plot and optionally save rewards
    if reward_logger_cb.rewards:
        plot_rewards(reward_logger_cb.rewards, save_path=log_dir / "training_rewards.png")

    return model


def evaluate(
    model_path: pathlib.Path,
    env_factory: Callable[[], gym.Env],
    episodes: int = 5,
    render: bool = False,
) -> None:
    """
    Load a trained model and run it for a number of episodes.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to the saved ``.zip`` model file.
    env_factory : Callable[[], gym.Env]
        Factory that creates a fresh environment instance.
    episodes : int, optional
        Number of evaluation episodes (default 5).
    render : bool, optional
        If ``True``, call ``env.render()`` each step (requires a renderable env).
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = env_factory()
    model = PPO.load(str(model_path), env=env)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            if render:
                env.render()
        print(f"[EVAL] Episode {ep + 1}/{episodes} – Steps: {step} – Total Reward: {total_reward:.2f}")

    env.close()


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Dynamic Multi‑Agent Traffic Light Control demo (train / evaluate)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub‑command to run")

    # ------------------------------------------------------------------- #
    # Train sub‑command
    # ------------------------------------------------------------------- #
    train_parser = subparsers.add_parser("train", help="Train a PPO agent")
    train_parser.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Total number of environment steps for training (default: 500k).",
    )
    train_parser.add_argument(
        "--envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1).",
    )
    train_parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("models/ppo_traffic.zip"),
        help="File path to store the trained model.",
    )
    train_parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=pathlib.Path("logs"),
        help="Directory for training logs and checkpoints.",
    )

    # ------------------------------------------------------------------- #
    # Evaluate sub‑command
    # ------------------------------------------------------------------- #
    eval_parser = subparsers.add_parser("eval", help="Run a trained model")
    eval_parser.add_argument(
        "--model",
        type=pathlib.Path,
        required=True,
        help="Path to the trained ``.zip`` model file.",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5).",
    )
    eval_parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation (if supported).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    env_factory = _import_traffic_env()

    if args.command == "train":
        train(
            env_factory=env_factory,
            total_timesteps=args.steps,
            model_path=args.output,
            log_dir=args.log_dir,
            n_envs=args.envs,
        )
    elif args.command == "eval":
        evaluate(
            model_path=args.model,
            env_factory=env_factory,
            episodes=args.episodes,
            render=args.render,
        )
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    # Guard to allow ``python -m src.main`` as well as direct execution.
    main()
