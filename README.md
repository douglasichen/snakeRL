# 🐍 snakeRL

A reinforcement learning (RL) project where an AI agent learns to play the classic Snake game using **Deep Q-Learning (DQN)** with PyTorch.

---

## 📦 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/douglasichen/snakeRL.git
   cd snakeRL
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n snakeRL python=3.9
   conda activate snakeRL
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


⸻

▶️ Run Training

To start training the agent:

```bash
python agent.py
```

This will:
	•	Launch the Snake game environment (game.py).
	•	Train the agent with experience replay and epsilon-greedy policy.
	•	Continuously update plots of score and mean score (helper.py).
	•	Save the trained model to ./model/model.pth.

⸻

📁 Project Structure

snakeRL/
│── agent.py        # Defines the RL Agent and training loop
│── game.py         # Snake game environment (pygame-based)
│── helper.py       # Real-time plotting of training progress
│── model.py        # Neural network (Q-Network) and trainer
│── model/
│   └── model.pth   # Saved trained model
│── README.md       # Project documentation


⸻

🧠 How It Works

### State Representation
The agent observes the game as an 11-dimensional binary state vector that captures the current game situation:

```python
state = [
    # Danger detection (3 values)
    danger_straight,    # Boolean: collision ahead in current direction
    danger_right,       # Boolean: collision to the right
    danger_left,        # Boolean: collision to the left
    
    # Movement direction (4 values)
    dir_l,             # Boolean: moving left
    dir_r,             # Boolean: moving right
    dir_u,             # Boolean: moving up
    dir_d,             # Boolean: moving down
    
    # Food location (4 values)
    food_left,         # Boolean: food is to the left
    food_right,        # Boolean: food is to the right
    food_up,           # Boolean: food is above
    food_down          # Boolean: food is below
]
```

**State Vector Details:**
- **Danger Detection**: Uses relative positioning based on snake's current direction
- **Movement Direction**: One-hot encoding of current heading (only one value is `True`)
- **Food Location**: Relative positioning of food compared to snake head
- **Advantages**: Invariant to absolute positions, generalizes across scenarios

### Neural Network Architecture
- **Input Layer**: 11 neurons (matches state vector dimension)
- **Hidden Layer**: 256 neurons with ReLU activation
- **Output Layer**: 3 neurons (Q-values for each action)

### Action Space
The agent can choose from 3 actions:
```python
# Action vector format: [straight, right, left]
[1, 0, 0]  # Continue straight (no turn)
[0, 1, 0]  # Turn right
[0, 0, 1]  # Turn left
```

**Action Logic:**
- Actions are relative to current direction
- Clockwise direction array: `[RIGHT, DOWN, LEFT, UP]`
- Right turn: `(current_index + 1) % 4`
- Left turn: `(current_index - 1) % 4`

### Training Process
- **Algorithm**: Q-Learning with discount factor `γ = 0.9`
- **Experience Replay**: Buffer size of 100,000 experiences
- **Batch Training**: 1,000 samples per batch
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate `lr=0.001`
- **Exploration**: Epsilon-greedy policy (ε = 80 - n_games)

### Reward Structure
- **+10**: Successfully eating food
- **-10**: Hitting wall or self (game over)
- **0**: Regular movement (implicit)

### Training Modes
1. **Short Memory**: Immediate training after each action
2. **Long Memory**: Batch training from experience replay buffer

### Game State Details
- **Grid Dimensions**: 640x480 pixels (32x24 blocks of 20x20 pixels each)
- **Block Size**: 20 pixels
- **Speed**: 4000ms per frame (slow for training)
- **Snake Representation**: List of Point objects `[head, body_segment_1, body_segment_2, ...]`
- **Point Structure**: `Point = namedtuple('Point', 'x, y')`

### State Representation Advantages
- **Relative Positioning**: State is invariant to absolute snake position
- **Binary Encoding**: Simple, interpretable representation
- **Comprehensive Information**: Captures immediate dangers, food location, and movement context
- **Compact Design**: Only 11 dimensions needed while maintaining efficiency

### Limitations & Future Improvements
- **Limited Vision**: Only immediate surroundings considered
- **Binary Granularity**: No distance information (only direction)
- **No History**: State doesn't include previous actions
- **Potential Enhancements**: Add snake length, distance to food, danger distances, or use convolutional approaches

⸻

📊 Visualization

Training progress is plotted in real-time:
- Blue: game scores
- Orange: mean score

⸻
