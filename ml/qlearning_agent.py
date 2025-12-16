# ml/final_q_agent.py
import numpy as np
from game import Agent
from pacman import Directions

class FinalQAgent(Agent):
    def __init__(self, alpha=0.05, gamma=0.9, epsilon=0.5):
        super().__init__()
        # Convert string args to floats
        if isinstance(alpha, str): alpha = float(alpha)
        if isinstance(gamma, str): gamma = float(gamma)
        if isinstance(epsilon, str): epsilon = float(epsilon)
        
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.weights = np.zeros(3)  # 3 simple features
    
    def get_features(self, state, action):
        """Simple features that make sense"""
        # Current Pac-Man position
        pac_x, pac_y = state.getPacmanPosition()
        
        # Where will Pac-Man be after this action?
        if action == Directions.NORTH: pac_y += 1
        elif action == Directions.SOUTH: pac_y -= 1
        elif action == Directions.EAST: pac_x += 1
        elif action == Directions.WEST: pac_x -= 1
        
        # Game state
        food = state.getFood().asList()
        ghosts = state.getGhostPositions()
        
        # FEATURE 1: Will this action eat food? (1 if yes, 0 if no)
        will_eat = 1.0 if (pac_x, pac_y) in food else 0.0
        
        # FEATURE 2: How close to the nearest food? (1.0 = on food, 0.0 = far)
        if food:
            distances = [abs(pac_x - x) + abs(pac_y - y) for x, y in food]
            closest = min(distances)
            food_score = 1.0 / (1.0 + closest)  # 1.0 if on food, smaller if far
        else:
            food_score = 0.0
        
        # FEATURE 3: Ghost danger (0 = safe, 1 = dangerous)
        danger = 0.0
        for gx, gy in ghosts:
            dist = abs(pac_x - gx) + abs(pac_y - gy)
            if dist == 0:  # Would collide with ghost
                danger = 1.0
            elif dist == 1:  # Very close to ghost
                danger = 0.7
            elif dist == 2:  # Somewhat close
                danger = 0.3
        
        return np.array([will_eat, food_score, danger])
    
    def getQValue(self, state, action):
        """Q(s,a) = weights • features"""
        features = self.get_features(state, action)
        return np.dot(self.weights, features)
    
    def getAction(self, state):
        """Choose action using epsilon-greedy"""
        legal = state.getLegalPacmanActions()
        if not legal:
            return Directions.STOP
        
        # epsilon-greedy: random action ε% of the time
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
        
        # Otherwise, choose best action
        q_values = [self.getQValue(state, a) for a in legal]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(legal, q_values) if q == max_q]
        return np.random.choice(best_actions)
    
    def update(self, state, action, reward, next_state):
        """Q-learning update rule"""
        # Best Q-value for next state
        legal_next = next_state.getLegalPacmanActions()
        if legal_next:
            next_q = max(self.getQValue(next_state, a) for a in legal_next)
        else:
            next_q = 0
        
        # TD error
        current_q = self.getQValue(state, action)
        td_error = reward + self.gamma * next_q - current_q
        
        if reward > 0:
            # Learn MORE from positive experiences
            learn_rate = self.alpha * 1.5
        else:
            # Learn LESS from negative experiences  
            learn_rate = self.alpha * 0.7
        
        # Get features
        features = self.get_features(state, action)
        
        # Update weights
        self.weights += learn_rate * td_error * features
        
        # === SOFT, GRADUAL CLIPPING ===
        # Instead of hard ±20 limits, use gradual decay
        for i in range(len(self.weights)):
            if self.weights[i] > 15:
                self.weights[i] = 15 + (self.weights[i] - 15) * 0.9  # Slow growth above 15
            elif self.weights[i] < -15:
                self.weights[i] = -15 + (self.weights[i] + 15) * 0.9  # Slow decay below -15

    def get_reward(self, state, action, next_state):
        """More balanced reward function"""
        reward = 0
        
        # === POSITIVE REWARDS (Encourage good behavior) ===
        
        # 1. BASE SURVIVAL REWARD (every step alive is good)
        reward += 2
        
        # 2. FOOD REWARDS (your suggestion - critical!)
        score_diff = next_state.getScore() - state.getScore()
        if score_diff == 10:  # Ate regular food
            reward += 15  # Consistent food reward
        
        # 3. GHOST EATING BONUS (your suggestion)
        elif score_diff == 200:  # Ate scared ghost
            reward += 250  # Big bonus for eating ghost
        
        # === TERMINAL REWARDS ===
        if next_state.isWin():
            return 500  # Win reward
        
        if next_state.isLose():
            return -100  # Loss penalty
        
        # === STRATEGIC REWARDS ===
        pacman_pos = state.getPacmanPosition()
        
        # Bonus for being near food (encourages exploration)
        food = state.getFood().asList()
        if food:
            dists = [abs(pacman_pos[0] - x) + abs(pacman_pos[1] - y) for x, y in food]
            if min(dists) <= 2:  # Close to food
                reward += 3
        
        # Penalty for being near active ghosts
        for ghost in state.getGhostStates():
            if ghost.scaredTimer == 0:  # Active ghost
                gx, gy = ghost.getPosition()
                dist = abs(pacman_pos[0] - gx) + abs(pacman_pos[1] - gy)
                if dist == 1:
                    reward -= 10
                elif dist == 2:
                    reward -= 3
        
        return reward