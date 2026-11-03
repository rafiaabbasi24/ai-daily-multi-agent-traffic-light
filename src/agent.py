import os
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor


class MultiAgentPPO:
    """
    Multi‑agent wrapper around Stable‑Baselines3 PPO.

    This class manages an independent PPO model for each agent in a
    multi‑agent environment.  The environment must follow the
    ``gymnasium`` API and return observations, rewards, dones and infos as
    dictionaries keyed by the agent identifiers.

    Example
    -------
    >>> import gymnasium as gym
    >>> from src.agent import MultiAgentPPO
    >>> env = gym.make("YourMultiAgentEnv-v0")
    >>> agents = list(env.agent_ids)          # e.g. ["tl_0", "tl_1"]
    >>> ma_ppo = MultiAgentPPO(env_fn=lambda: env, agent_ids=agents)
    >>> ma_ppo.train(total_timesteps=100_000)
    >>> rewards = ma_ppo.evaluate(num_episodes=5)
    >>> ma_ppo.save("saved_models")
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        agent_ids: Iterable[str],
        policy_kwargs: Dict[str, Any] | None = None,
        device: Union[str, torch.device] = "auto",
        seed: int | None = None,
        log_dir: str | os.PathLike = "logs",
    ) -> None:
        """
        Initialise the multi‑agent PPO trainer.

        Parameters
        ----------
        env_fn :
            Callable that returns a fresh instance of the environment.
            The function is called once per agent to build a vectorised
            environment.
        agent_ids :
            Iterable of unique identifiers for each traffic‑light agent.
        policy_kwargs :
            Keyword arguments passed to the PPO policy constructor.
        device :
            Device on which to run the models (``"cpu"``, ``"cuda"``, or ``"auto"``).
        seed :
            Random seed for reproducibility.
        log_dir :
            Directory where TensorBoard logs and intermediate results are stored.
        """
        self.agent_ids: List[str] = list(agent_ids)
        if len(self.agent_ids) == 0:
            raise ValueError("At least one agent id must be provided.")

        self.device = device
        self.seed = seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._policy_kwargs = policy_kwargs or {}
        self._env_fns: Dict[str, Callable[[], VecEnv]] = {}
        self.models: Dict[str, PPO] = {}

        # Create a vectorised environment and a PPO model for each agent
        for aid in self.agent_ids:
            # Each agent gets its own independent environment instance.
            # The environment must be able to handle a single active agent.
            # If the original env already supports multi‑agent, we wrap it
            # with a filter that only exposes the observation/reward for the
            # current agent.
            def make_agent_env(aid: str = aid) -> VecEnv:
                # Use DummyVecEnv for simplicity; can be swapped for SubprocVecEnv.
                env = env_fn()
                # Seed the environment if a seed is supplied.
                if self.seed is not None:
                    env.reset(seed=self.seed)
                # Wrap to monitor statistics.
                vec_env = DummyVecEnv([lambda: env])
                return VecMonitor(vec_env)

            self._env_fns[aid] = make_agent_env
            self.models[aid] = PPO(
                "MlpPolicy",
                self._env_fns[aid](),
                device=self.device,
                seed=self.seed,
                verbose=0,
                tensorboard_log=str(self.log_dir / f"tensorboard_{aid}"),
                **self._policy_kwargs,
            )

    def train(
        self,
        total_timesteps: int,
        callback: BaseCallback | None = None,
        log_interval: int = 10,
    ) -> None:
        """
        Train each agent's PPO model.

        Parameters
        ----------
        total_timesteps :
            Number of timesteps to train each model.
        callback :
            Optional Stable‑Baselines3 callback.
        log_interval :
            Frequency (in thousands of timesteps) at which progress is printed.
        """
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer.")

        for aid, model in self.models.items():
            print(f"[{aid}] Starting training for {total_timesteps:,} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=log_interval,
            )
            print(f"[{aid}] Training completed.")

    def evaluate(
        self,
        env_fn: Callable[[], gym.Env],
        num_episodes: int = 5,
        render: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Run deterministic evaluation episodes and collect rewards.

        Parameters
        ----------
        env_fn :
            Callable that returns a fresh environment for evaluation.
        num_episodes :
            Number of episodes to run.
        render :
            If True, calls ``env.render()`` at each step (useful for debugging).

        Returns
        -------
        dict
            Mapping from agent id to a list of episode returns.
        """
        if num_episodes <= 0:
            raise ValueError("num_episodes must be a positive integer.")

        episode_rewards: Dict[str, List[float]] = {aid: [] for aid in self.agent_ids}

        for ep in range(num_episodes):
            env = env_fn()
            if self.seed is not None:
                obs, _ = env.reset(seed=self.seed + ep)
            else:
                obs, _ = env.reset()
            done = {aid: False for aid in self.agent_ids}
            cumulative: Dict[str, float] = {aid: 0.0 for aid in self.agent_ids}

            while not all(done.values()):
                # Compute actions for each active agent
                actions = {}
                for aid in self.agent_ids:
                    if done[aid]:
                        continue
                    # Extract the observation slice for the current agent.
                    # The environment is assumed to return a dict of observations.
                    agent_obs = obs[aid]
                    # Stable‑Baselines3 expects a batch dimension.
                    action, _ = self.models[aid].predict(
                        np.array([agent_obs]), deterministic=True
                    )
                    actions[aid] = action[0]

                # Step the environment with the joint action dict
                obs, rewards, terminated, truncated, info = env.step(actions)

                # Update termination flags
                for aid in self.agent_ids:
                    if done[aid]:
                        continue
                    done[aid] = terminated.get(aid, False) or truncated.get(aid, False)
                    cumulative[aid] += rewards.get(aid, 0.0)

                if render:
                    env.render()

            for aid in self.agent_ids:
                episode_rewards[aid].append(cumulative[aid])

            env.close()

        return episode_rewards

    def act(self, observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compute a deterministic action for each agent given current observations.

        Parameters
        ----------
        observations :
            Mapping from agent id to its observation array.

        Returns
        -------
        dict
            Mapping from agent id to the selected action.
        """
        actions: Dict[str, Any] = {}
        for aid, obs in observations.items():
            if aid not in self.models:
                raise KeyError(f"Agent id '{aid}' not recognised.")
            # Predict deterministic action
            action, _ = self.models[aid].predict(np.array([obs]), deterministic=True)
            actions[aid] = action[0]
        return actions

    def save(self, directory: Union[str, os.PathLike]) -> None:
        """
        Save all PPO models and a meta‑information file.

        Parameters
        ----------
        directory :
            Destination folder. It will be created if it does not exist.
        """
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save each model in its own sub‑folder
        for aid, model in self.models.items():
            agent_dir = save_path / aid
            agent_dir.mkdir(parents=True, exist_ok=True)
            model_path = agent_dir / "ppo.zip"
            model.save(str(model_path))
            # Also store the environment spec if needed (placeholder)
            with open(agent_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"agent_id": aid}, f)

        # Store a top‑level manifest
        manifest = {
            "agent_ids": self.agent_ids,
            "policy_kwargs": self._policy_kwargs,
            "device": str(self.device),
            "seed": self.seed,
        }
        with open(save_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"All models saved to {save_path}")

    def load(self, directory: Union[str, os.PathLike]) -> None:
        """
        Load PPO models from a directory created by :meth:`save`.

        Parameters
        ----------
        directory :
            Path to the folder containing saved models.
        """
        load_path = Path(directory)
        manifest_path = load_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Validate manifest consistency
        saved_ids = set(manifest.get("agent_ids", []))
        expected_ids = set(self.agent_ids)
        if saved_ids != expected_ids:
            raise ValueError(
                f"Mismatch between saved agents {saved_ids} and expected agents {expected_ids}"
            )

        # Load each model
        for aid in self.agent_ids:
            model_path = load_path / aid / "ppo.zip"
            if not model_path.is_file():
                raise FileNotFoundError(f"Model file for agent '{aid}' not found at {model_path}")
            self.models[aid] = PPO.load(str(model_path), env=self._env_fns[aid](), device=self.device)

        print(f"All models loaded from {load_path}")

    def plot_learning_curves(
        self,
        log_dir: Union[str, os.PathLike] | None = None,
        save_path: Union[str, os.PathLike] | None = None,
    ) -> None:
        """
        Plot the learning curves (episode reward vs. timesteps) for each agent.

        Parameters
        ----------
        log_dir :
            Directory containing the TensorBoard log files. If ``None``,
            the directory passed at construction time is used.
        save_path :
            If provided, the figure is saved to this path; otherwise it is shown.
        """
        import pandas as pd
        from stable_baselines3.common.monitor import load_results

        base_dir = Path(log_dir) if log_dir is not None else self.log_dir

        plt.figure(figsize=(10, 6))
        for aid in self.agent_ids:
            tb_dir = base_dir / f"tensorboard_{aid}"
            if not tb_dir.is_dir():
                warnings.warn(f"No TensorBoard logs found for agent '{aid}' in {tb_dir}")
                continue
            try:
                df = load_results(tb_dir)
                # The monitor file stores cumulative timesteps in the "l" column.
                df["timestep"] = np.cumsum(df["l"])
                plt.plot(df["timestep"], df["r"], label=aid)
            except Exception as exc:
                warnings.warn(f"Failed to load results for agent '{aid}': {exc}")

        plt.xlabel("Timesteps")
        plt.ylabel("Episode Return")
        plt.title("Learning Curves per Traffic‑Light Agent")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Learning curve saved to {save_path}")
        else:
            plt.show()


__all__ = ["MultiAgentPPO"]