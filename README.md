# Dynamic Multi‑Agent Traffic Light Control

## Overview

**Dynamic Multi‑Agent Traffic Light Control** is a reinforcement learning (RL) framework for optimizing traffic signal timings in a simulated urban environment. The project demonstrates how multiple agents—each controlling an individual traffic light—can learn coordinated policies that reduce overall vehicle waiting time, improve traffic flow, and adapt to varying traffic patterns.

Key features:

- **Multi‑agent environment** built on **Gymnasium** with a realistic road network.
- **Deep RL algorithms** from **Stable‑Baselines3** (e.g., PPO, A2C) with PyTorch back‑ends.
- **Modular architecture** allowing easy swapping of agents, reward functions, and traffic scenarios.
- **Extensive visualisation** using Matplotlib for traffic density, queue lengths, and learned policies.
- **Reproducible experiments** via configurable YAML files and random seed handling.

## Project Structure

```
Dynamic-MultiAgent-Traffic-Light-Control/
├── configs/
│   └── default.yaml          # Default experiment configuration
├── env/
│   ├── __init__.py
│   └── traffic_env.py        # Gymnasium environment definition
├── agents/
│   ├── __init__.py
│   └── rl_agent.py           # Wrapper around Stable‑Baselines3 agents
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Simple logging helper
│   └── plot.py               # Matplotlib visualisation utilities
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python **3.9** or newer
- pip (>=23.0)

### Step‑by‑step

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Dynamic-MultiAgent-Traffic-Light-Control.git
   cd Dynamic-MultiAgent-Traffic-Light-Control
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` contains:

   ```
   torch>=2.0
   gymnasium>=0.28
   numpy>=1.24
   matplotlib>=3.7
   stable-baselines3>=2.2
   pyyaml>=6.0
   tqdm>=4.65
   ```

4. **Verify the installation**

   ```bash
   python -c "import torch, gymnasium, stable_baselines3; print('All imports OK')"
   ```

## Configuration

All experiment parameters are stored in YAML files under `configs/`. The default configuration (`configs/default.yaml`) includes:

```yaml
seed: 42
env:
  num_intersections: 4
  max_steps: 500
  traffic_pattern: uniform   # options: uniform, rush_hour, random
  vehicle_spawn_rate: 0.2
agent:
  algorithm: PPO
  policy: MlpPolicy
  learning_rate: 3e-4
  gamma: 0.99
  n_steps: 2048
  batch_size: 64
  total_timesteps: 1_000_000
logging:
  log_dir: logs/
  checkpoint_interval: 100_000
  eval_interval: 50_000
  eval_episodes: 20
```

You can create custom configurations and pass them to the scripts via the `--config` flag.

## Usage

### Training

```bash
python train.py --config configs/default.yaml
```

- **Outputs**
  - Model checkpoints saved under `logs/checkpoints/`.
  - TensorBoard logs under `logs/tensorboard/`.
  - CSV logs with episode statistics under `logs/csv/`.

### Evaluation

```bash
python evaluate.py --model logs/checkpoints/best_model.zip --episodes 30
```

- The script loads a trained model, runs the specified number of episodes, and prints aggregate metrics (average waiting time, total reward, etc.).
- A plot of queue lengths over time is saved to `logs/evaluation/queue_plot.png`.

### Visualisation

You can visualise a single episode interactively:

```bash
python evaluate.py --model logs/checkpoints/best_model.zip --render
```

The environment renders a Matplotlib animation showing vehicle movements and traffic‑light phases.

## Development

### Running Tests

The project includes a minimal test suite based on `pytest`. To run the tests:

```bash
pip install pytest
pytest tests/
```

### Extending the Environment

- **Add new traffic patterns**: modify `env/traffic_env.py` and expose a new `traffic_pattern` option in the config.
- **Custom reward functions**: subclass `env/traffic_env.TrafficEnv` and override `_compute_reward`.

### Adding New RL Algorithms

Stable‑Baselines3 supports many algorithms. To use a different one, change the `algorithm` field in the config (e.g., `A2C`, `SAC`). Ensure the corresponding policy class is compatible (e.g., `MlpPolicy`).

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Citation

If you use this code in academic work, please cite:

```
@software{DynamicMultiAgentTrafficLight2026,
  author = {Your Name and Contributors},
  title = {Dynamic Multi‑Agent Traffic Light Control},
  year = {2026},
  url = {https://github.com/yourusername/Dynamic-MultiAgent-Traffic-Light-Control},
  license = {MIT}
}
```

--- 

*Happy experimenting! 🚦*