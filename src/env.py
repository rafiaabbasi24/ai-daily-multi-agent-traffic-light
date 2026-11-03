import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional


class TrafficLightEnv(gym.Env):
    """
    A custom multi‑agent Gymnasium environment that simulates a simple traffic network
    consisting of several intersections, each controlled by an independent traffic light.
    The goal of each agent (intersection) is to minimise the total queue length of cars
    waiting on its incoming lanes.

    The environment is intentionally lightweight yet realistic enough for reinforcement
    learning experiments. It can be used with Stable‑Baselines3 or any other RL library
    that supports dictionary‑based observation and action spaces.

    **Observation Space**
        ``Dict`` mapping each intersection id (``str``) to a ``Box`` of shape ``(4,)``:
        ``[q_north, q_south, q_east, q_west]`` where each entry is the current queue
        length on the corresponding incoming lane (integer in ``[0, max_queue]``).

    **Action Space**
        ``Dict`` mapping each intersection id to a ``Discrete(2)``:
        ``0`` – green for north‑south traffic, red for east‑west;
        ``1`` – green for east‑west traffic, red for north‑south.

    **Reward**
        Negative sum of queue lengths for the intersection (i.e. ``-sum(queues)``).  This
        encourages agents to keep queues short.

    **Termination**
        The episode ends after ``max_steps`` steps or when all queues are empty.
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        num_intersections: int = 4,
        max_queue: int = 10,
        arrival_rate: float = 0.2,
        departure_rate: int = 1,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialise the environment.

        Args:
            num_intersections: Number of intersections (agents) in the grid.
            max_queue: Maximum number of cars that can wait on a lane.
            arrival_rate: Probability of a new car arriving on each lane at each step.
            departure_rate: Number of cars that can leave a green lane per step.
            max_steps: Horizon of an episode.
            seed: Random seed for reproducibility.
        """
        super().__init__()

        self.num_intersections = num_intersections
        self.max_queue = max_queue
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate
        self.max_steps = max_steps

        # Create a deterministic RNG; Gymnasium's seeding utilities are used.
        self._rng = np.random.default_rng(seed)

        # Define the agents' identifiers.
        self.agents = [f"intersection_{i}" for i in range(self.num_intersections)]

        # Observation space: per‑agent queue lengths for 4 directions.
        obs_space = spaces.Box(
            low=0,
            high=self.max_queue,
            shape=(4,),
            dtype=np.int32,
        )
        self.observation_space = spaces.Dict({agent: obs_space for agent in self.agents})

        # Action space: 0 – NS green, 1 – EW green.
        act_space = spaces.Discrete(2)
        self.action_space = spaces.Dict({agent: act_space for agent in self.agents})

        # Internal state containers.
        self.queues: Dict[str, np.ndarray] = {}
        self.current_step: int = 0
        self._episode_ended: bool = False

        # Rendering helpers.
        self._fig = None
        self._ax = None

        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to an initial state.

        Returns:
            A tuple ``(observation, info)`` where ``observation`` is a dictionary
            mapping each agent to its initial queue vector.
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Initialise all queues to zero.
        self.queues = {
            agent: np.zeros(4, dtype=np.int32) for agent in self.agents
        }  # order: N, S, E, W
        self.current_step = 0
        self._episode_ended = False

        observation = {agent: self.queues[agent].copy() for agent in self.agents}
        info = {"step": self.current_step}
        return observation, info

    def step(
        self,
        action: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Execute one time‑step of the environment's dynamics.

        Args:
            action: Dictionary mapping each agent id to its chosen action (0 or 1).

        Returns:
            A tuple ``(obs, reward, terminated, truncated, info)`` where each element
            is a dictionary keyed by agent id.
        """
        if self._episode_ended:
            raise RuntimeError("Attempted to call step() after the episode terminated. Call reset().")

        # Validate actions.
        for agent, act in action.items():
            if agent not in self.agents:
                raise ValueError(f"Unknown agent '{agent}' in action dict.")
            if act not in (0, 1):
                raise ValueError(f"Invalid action {act} for agent '{agent}'. Must be 0 or 1.")

        # 1. Departures based on current phase.
        for agent, act in action.items():
            queues = self.queues[agent]
            # Determine which lanes are green.
            if act == 0:  # NS green
                green_idxs = (0, 1)  # North, South
            else:  # EW green
                green_idxs = (2, 3)  # East, West

            for idx in green_idxs:
                departure = min(self.departure_rate, queues[idx])
                queues[idx] -= departure

        # 2. New arrivals (independent Bernoulli per lane).
        for agent in self.agents:
            arrivals = self._rng.binomial(1, self.arrival_rate, size=4)
            self.queues[agent] = np.clip(
                self.queues[agent] + arrivals,
                0,
                self.max_queue,
            )

        # 3. Compute observations, rewards, termination flags.
        obs: Dict[str, np.ndarray] = {}
        reward: Dict[str, float] = {}
        terminated: Dict[str, bool] = {}
        truncated: Dict[str, bool] = {}
        info: Dict[str, Any] = {}

        total_queue = 0
        for agent in self.agents:
            q = self.queues[agent]
            obs[agent] = q.copy()
            reward[agent] = -float(q.sum())  # minimise total waiting cars
            terminated[agent] = False  # per‑agent termination not used here
            truncated[agent] = False
            info[agent] = {}
            total_queue += q.sum()

        self.current_step += 1

        # Global termination conditions.
        if self.current_step >= self.max_steps or total_queue == 0:
            self._episode_ended = True
            for agent in self.agents:
                terminated[agent] = True

        # Global info (same for all agents).
        for agent in self.agents:
            info[agent]["step"] = self.current_step
            info[agent]["total_queue"] = total_queue

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state of the traffic network using Matplotlib.

        Returns:
            If ``mode`` is ``rgb_array``, returns an ``np.ndarray`` image. Otherwise,
            displays the figure and returns ``None``.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(4, 4))
        else:
            self._ax.clear()

        grid_size = int(np.ceil(np.sqrt(self.num_intersections)))
        for idx, agent in enumerate(self.agents):
            row = idx // grid_size
            col = idx % grid_size
            x = col
            y = grid_size - 1 - row

            # Draw a square for the intersection.
            square = plt.Rectangle((x, y), 1, 1, edgecolor="k", facecolor="lightgray")
            self._ax.add_patch(square)

            # Show current phase as an arrow.
            # Retrieve last action; if none (first step) default to NS green.
            # For simplicity we store last actions in info; fallback to 0.
            last_action = 0
            if hasattr(self, "_last_action"):
                last_action = self._last_action.get(agent, 0)

            if last_action == 0:
                # NS green: vertical arrow
                self._ax.arrow(x + 0.5, y + 0.2, 0, 0.6, head_width=0.1, head_length=0.1, fc="g", ec="g")
                self._ax.arrow(x + 0.5, y + 0.8, 0, -0.6, head_width=0.1, head_length=0.1, fc="g", ec="g")
            else:
                # EW green: horizontal arrow
                self._ax.arrow(x + 0.2, y + 0.5, 0.6, 0, head_width=0.1, head_length=0.1, fc="g", ec="g")
                self._ax.arrow(x + 0.8, y + 0.5, -0.6, 0, head_width=0.1, head_length=0.1, fc="g", ec="g")

            # Plot queue lengths as text.
            q = self.queues[agent]
            directions = ["N", "S", "E", "W"]
            for i, dir_label in enumerate(directions):
                offset = {
                    0: (0.5, 0.9),  # N
                    1: (0.5, 0.1),  # S
                    2: (0.9, 0.5),  # E
                    3: (0.1, 0.5),  # W
                }[i]
                self._ax.text(
                    x + offset[0],
                    y + offset[1],
                    f"{dir_label}:{q[i]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

        self._ax.set_xlim(0, grid_size)
        self._ax.set_ylim(0, grid_size)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_aspect("equal")
        self._fig.canvas.draw()
        plt.pause(1.0 / self.metadata["render_fps"])

        return None

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def seed(self, seed: Optional[int] = None) -> list[int]:
        """
        Set the seed for this environment's random number generator.

        Returns:
            A list containing the used seed.
        """
        self._rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    # Helper to store the last action for rendering.
    def _store_last_action(self, action: Dict[str, int]) -> None:
        self._last_action = action.copy()
    
    # Overriding step to store actions for rendering.
    def step(
        self,
        action: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        self._store_last_action(action)
        return super().step(action)  # type: ignore[misc]