import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, **env_kwargs: Any) -> Callable[[], gym.Env]:
    """
    Return a callable that creates a new environment instance.
    This wrapper is required by Stable‑Baselines3 when using vectorized environments.

    Parameters
    ----------
    env_id : str
        Identifier of the environment. It can be a Gymnasium registration ID
        or a fully‑qualified class path.
    **env_kwargs : dict
        Additional keyword arguments passed to the environment constructor.

    Returns
    -------
    Callable[[], gym.Env]
        A function that creates a fresh environment when called.
    """
    def _init() -> gym.Env:
        # Try to create the environment via gym.make first.
        try:
            env = gym.make(env_id, **env_kwargs)
        except Exception as e_gym:
            # Fall back to importing a custom environment class.
            try:
                # Expected pattern: "module:ClassName"
                module_path, class_name = env_id.split(":")
                module = __import__(module_path, fromlist=[class_name])
                env_cls = getattr(module, class_name)
                env = env_cls(**env_kwargs)
            except Exception as e_custom:
                raise RuntimeError(
                    f"Unable to create environment '{env_id}'. "
                    f"Gym error: {e_gym}; Custom import error: {e_custom}"
                ) from e_custom
        return env

    return _init


class RewardLoggerCallback(BaseCallback):
    """
    Custom callback that records episode rewards and lengths.
    The data is stored in ``self.locals`` and can be accessed after training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        # ``infos`` is a list of dicts, one per environment.
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def get_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays of recorded rewards and lengths.

        Returns
        -------
        rewards : np.ndarray
            Recorded episode rewards.
        lengths : np.ndarray
            Recorded episode lengths.
        """
        return np.array(self.episode_rewards), np.array(self.episode_lengths)


def plot_training_curve(
    rewards: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Training Reward Curve",
) -> None:
    """
    Plot and optionally save the training reward curve.

    Parameters
    ----------
    rewards : np.ndarray
        Array of episode rewards.
    save_path : Path | None, optional
        If provided, the figure is saved to this location.
    title : str, optional
        Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Training curve saved to {save_path}")
    plt.close()


# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #
def train(
    env_id: str,
    total_timesteps: int,
    seed: int = 42,
    log_dir: str = "./logs",
    save_dir: str = "./models",
    eval_env_id: Optional[str] = None,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    checkpoint_freq: int = 50_000,
    num_envs: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    device: str = "auto",
) -> Tuple[PPO, RewardLoggerCallback]:
    """
    Execute the training loop.

    Parameters
    ----------
    env_id : str
        Identifier for the training environment.
    total_timesteps : int
        Number of environment steps to train for.
    seed : int, default 42
        Random seed for reproducibility.
    log_dir : str, default "./logs"
        Directory where TensorBoard logs will be stored.
    save_dir : str, default "./models"
        Directory where model checkpoints will be saved.
    eval_env_id : str | None, default None
        Identifier for an optional evaluation environment. If ``None``,
        the training environment is reused for evaluation.
    eval_freq : int, default 10_000
        Frequency (in steps) of evaluation runs.
    n_eval_episodes : int, default 5
        Number of episodes per evaluation.
    checkpoint_freq : int, default 50_000
        Frequency (in steps) to save model checkpoints.
    num_envs : int, default 4
        Number of parallel environments for training.
    learning_rate : float, default 3e-4
        Learning rate for the optimizer.
    gamma : float, default 0.99
        Discount factor.
    device : str, default "auto"
        Device to use (``"cpu"``, ``"cuda"``, or ``"auto"``).

    Returns
    -------
    model : PPO
        The trained PPO model.
    reward_logger : RewardLoggerCallback
        Callback containing logged rewards.
    """
    # ------------------------------------------------------------------- #
    # Setup
    # ------------------------------------------------------------------- #
    _set_random_seed(seed)

    log_path = Path(log_dir).expanduser().resolve()
    save_path = Path(save_dir).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    # Configure the SB3 logger to write TensorBoard files.
    new_logger = configure(str(log_path), ["tensorboard", "stdout"])

    # ------------------------------------------------------------------- #
    # Environment creation
    # ------------------------------------------------------------------- #
    env_fns = [make_env(env_id, seed=seed + i) for i in range(num_envs)]
    if num_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)  # adds episode info to ``infos``

    # ------------------------------------------------------------------- #
    # Evaluation environment (optional)
    # ------------------------------------------------------------------- #
    eval_env = None
    if eval_env_id:
        eval_fns = [make_env(eval_env_id, seed=seed + 1000)]
        eval_env = DummyVecEnv(eval_fns)
        eval_env = VecMonitor(eval_env)

    # ------------------------------------------------------------------- #
    # Callbacks
    # ------------------------------------------------------------------- #
    reward_logger = RewardLoggerCallback(verbose=0)

    callbacks: List[BaseCallback] = [reward_logger]

    # Checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_path),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_cb)

    # Evaluation callback
    if eval_env is not None:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(save_path / "best_model"),
            log_path=str(log_path / "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)

    # Combine callbacks into a single one
    class CallbackList(BaseCallback):
        def __init__(self, callbacks: List[BaseCallback], verbose: int = 0):
            super().__init__(verbose)
            self.callbacks = callbacks

        def _on_training_start(self) -> None:
            for cb in self.callbacks:
                cb.set_training_env(self.training_env)
                cb.set_logger(self.logger)
                cb._on_training_start()

        def _on_step(self) -> bool:
            # Return False to abort training if any callback returns False.
            for cb in self.callbacks:
                if not cb._on_step():
                    return False
            return True

        def _on_training_end(self) -> None:
            for cb in self.callbacks:
                cb._on_training_end()

    callback = CallbackList(callbacks, verbose=1)

    # ------------------------------------------------------------------- #
    # Model definition
    # ------------------------------------------------------------------- #
    device_opt = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
        tensorboard_log=str(log_path),
        device=device_opt,
        seed=seed,
    )
    model.set_logger(new_logger)

    # ------------------------------------------------------------------- #
    # Training
    # ------------------------------------------------------------------- #
    logging.info("Starting training")
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)

    # Save final model
    final_model_path = save_path / "ppo_final.zip"
    model.save(str(final_model_path))
    logging.info(f"Final model saved to {final_model_path}")

    # ------------------------------------------------------------------- #
    # Plot rewards
    # ------------------------------------------------------------------- #
    rewards, _ = reward_logger.get_statistics()
    if rewards.size > 0:
        plot_path = log_path / "training_reward_curve.png"
        plot_training_curve(rewards, save_path=plot_path)

    # Clean up environments
    vec_env.close()
    if eval_env is not None:
        eval_env.close()

    return model, reward_logger


# --------------------------------------------------------------------------- #
# Argument parsing and entry point
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a dynamic multi‑agent traffic‑light control policy using PPO."
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Gymnasium environment ID or module path (e.g., 'my_pkg.envs:TrafficEnv').",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of environment steps to train for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs and plots.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory where model checkpoints will be stored.",
    )
    parser.add_argument(
        "--eval-env",
        type=str,
        default=None,
        help="Optional evaluation environment ID.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency in steps.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Number of episodes per evaluation.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint saving frequency in steps.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel training environments.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device on which to run the training.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = _build_parser()
    args = parser.parse_args()

    train(
        env_id=args.env,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        eval_env_id=args.eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=args.device,
    )


if __name__ == "__main__":
    main()
