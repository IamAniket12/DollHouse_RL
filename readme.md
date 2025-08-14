# Dollhouse Thermal Control

A reinforcement learning environment for thermal control in a two-floor dollhouse. Train RL agents to maintain comfortable temperatures while minimizing energy usage.

## 🚀 Quick Start

### 1. Create Environment & Install Dependencies

**Create a new conda environment:**
```bash
conda create -n dollhouse python=3.9
conda activate dollhouse
```

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

You need a CSV file with dollhouse sensor data to train the physics model (SINDy). The data should contain temperature and control measurements. It is available inside the Data folder.

### 3. Train an RL Agent

**Basic training with default settings:**
```bash
python train_rl_agent.py --data your_data.csv
```

**Custom training:**
```bash
python train_rl_agent.py --data your_data.csv --algorithm ppo --timesteps 5000000 --obs-preset compact
```

That's it! 🎉

## 🛠️ Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.8+ (recommended: 3.9)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/dollhouse-thermal-control.git
   cd dollhouse-thermal-control
   ```

2. **Create a dedicated conda environment:**
   ```bash
   conda create -n dollhouse python=3.9
   conda activate dollhouse
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python train_rl_agent.py --help
   ```

> **💡 Tip:** Always activate the environment before use: `conda activate dollhouse`

## 📋 Environment Overview

The environment simulates a two-floor dollhouse with:
- **Ground floor** and **top floor** temperatures
- **Controllable lights** (heating when ON)
- **Controllable windows** (for ventilation)
- **External temperature** (time-varying)
- **Temperature setpoints** (comfort zones)

**Goal:** Keep both floors comfortable while using minimal energy.

## 🎯 Training Options

### Algorithms
Choose your RL algorithm:
```bash
--algorithm ppo    # (recommended)
--algorithm a2c
--algorithm sac
--algorithm dqn
```

### Observation Presets
Choose what the agent observes:

```bash
--obs-preset minimal     # Just temperatures & setpoints (4 obs)
--obs-preset standard    # Basic set with controls (10 obs)  
--obs-preset extended    # Includes derived features (15 obs)
--obs-preset compact     # Efficient feature set (8 obs)
--obs-preset time_aware  # Time-focused observations (8 obs)
```

### Custom Observations
Or specify exactly what you want:
```bash
--observations "ground_temp,top_temp,heating_setpoint,cooling_setpoint"
```

### Performance Options
```bash
--n-envs 8           # Use 8 parallel environments (faster training)
--timesteps 10000000 # Train for 10M steps
--no-normalize       # Disable automatic normalization
```

## 📊 Available Observations

| Observation | Description |
|-------------|-------------|
| `ground_temp` | Ground floor temperature |
| `top_temp` | Top floor temperature |
| `external_temp` | Outside temperature |
| `ground_light` | Ground floor light state (0/1) |
| `top_light` | Top floor light state (0/1) |
| `ground_window` | Ground floor window state (0/1) |
| `top_window` | Top floor window state (0/1) |
| `heating_setpoint` | Minimum comfortable temperature |
| `cooling_setpoint` | Maximum comfortable temperature |
| `temp_difference` | Temperature difference between floors |
| `sin_hour` | Sine of hour (for daily patterns) |
| `cos_hour` | Cosine of hour (for daily patterns) |
| `total_lights_on` | Number of lights currently on |
| ... | [See full list with `--list-observations`] |

## 💡 Examples

### List all available options:
```bash
python train_rl_agent.py --data your_data.csv --list-observations
```

### Quick efficient training:
```bash
python train_rl_agent.py --data your_data.csv --obs-preset compact --n-envs 8 --timesteps 2000000
```

### High-performance training:
```bash
python train_rl_agent.py --data your_data.csv --algorithm ppo --obs-preset extended --n-envs 16 --timesteps 10000000
```

### Custom observations for research:
```bash
python train_rl_agent.py --data your_data.csv --observations "ground_temp_deviation,top_temp_deviation,sin_hour,cos_hour,total_lights_on"
```

### Training without normalization:
```bash
python train_rl_agent.py --data your_data.csv --obs-preset standard --no-normalize
```

## 📈 Monitoring Training

### WandB Integration (Recommended)
Training automatically logs to [Weights & Biases](https://wandb.ai/):
- Loss curves and learning progress
- Environment metrics (temperature, comfort violations, energy use)
- Episode rewards and statistics

### Disable WandB:
```bash
python train_rl_agent.py --data your_data.csv --no-wandb
```

## 🏠 Using the Environment Directly

```python
from dollhouse_env import DollhouseThermalEnv
from train_sindy_model import train_sindy_model

# Train physics model
sindy_model = train_sindy_model("your_data.csv")

# Create environment
env = DollhouseThermalEnv(
    sindy_model=sindy_model,
    custom_observations=["ground_temp", "top_temp", "heating_setpoint", "cooling_setpoint"]
)

# Use like any Gym environment
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random actions
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## 🔧 Advanced Configuration

### Environment Parameters
```python
env = DollhouseThermalEnv(
    sindy_model=sindy_model,
    
    # Episode settings
    episode_length=2880,        # 24 hours (30-sec timesteps)
    time_step_seconds=30,
    
    # Comfort settings
    heating_setpoint=22.0,      # °C
    cooling_setpoint=26.0,      # °C
    
    # Environment dynamics
    external_temp_pattern="sine",  # "fixed", "sine", "real_data", "random_walk"
    setpoint_pattern="schedule",   # "fixed", "schedule", "adaptive", "challenging"
    
    # Learning features
    use_reward_shaping=True,    # Accelerated learning
    random_start_time=True,     # Start episodes at random times
    
    # Custom observations
    custom_observations=["ground_temp", "top_temp", "sin_hour", "cos_hour"]
)
```



## 🎮 Testing Your Agent

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Load trained model
model = PPO.load("results/ppo_20241215_143022/logs/models/ppo_final_model")
vec_env = VecNormalize.load("results/ppo_20241215_143022/logs/models/vec_normalize.pkl", env)

# Test the agent
obs = vec_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
```

## 🤝 Contributing

This is an open-source project! Feel free to:
- Add new observation types
- Improve the reward function
- Add new environment patterns
- Enhance the training script


