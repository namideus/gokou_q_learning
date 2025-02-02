# Import necessary libraries
from flask import Flask, render_template, jsonify, request
import numpy as np
import random
import pickle
import os

# Initialize Flask application
app = Flask(__name__)

# Game configuration constants
BOARD_SIZE = 9        # Size of the game board (9x9)
WIN_CONDITION = 5     # Number of consecutive marks needed to win
Q_FILE = "ai_q_learning.pkl"  # File to store learned Q-tables

class QLearningAgent:
    """Reinforcement learning agent using Q-learning algorithm"""
    
    def __init__(self, player):
        """Initialize Q-learning agent with player-specific parameters"""
        self.q_table = {}          # State-action value store
        self.gamma = 0.9           # Discount factor for future rewards
        self.player = player       # 'X' or 'O'
        self.last_state = None     # Track previous state for opponent penalty
        self.last_action = None    # Track previous action for opponent penalty

        # Differentiated learning parameters for balanced competition
        if player == 'X':
            self.alpha = 0.08      # Slower learning rate for X
            self.epsilon = 0.15    # Higher exploration rate for X
        else:
            self.alpha = 0.12      # Faster learning rate for O
            self.epsilon = 0.05    # Lower exploration rate for O

    def get_state_key(self, board):
        """Convert board state to string key for Q-table lookup"""
        return ''.join(board.flatten())  # Flatten 2D array to 81-character string

    def choose_action(self, state_key, available_actions):
        """Select action using ε-greedy policy"""
        # Exploration: random action with probability epsilon
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: choose action with highest Q-value
        q_values = [self.q_table.get((state_key, action), 0) for action in available_actions]
        max_q = max(q_values) if q_values else 0  # Handle empty list
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions) if best_actions else None

    def update_q_value(self, state_key, action, reward, next_state_key):
        """Update Q-value using Bellman equation"""
        if action is None:
            return  # Skip invalid actions
            
        old_q = self.q_table.get((state_key, action), 0)
        next_actions = self.get_possible_actions(next_state_key)
        next_max_q = max([self.q_table.get((next_state_key, a), 0) for a in next_actions] or [0])
        
        # Q-learning formula: Q(s,a) += α * (reward + γ * maxQ(s',a') - Q(s,a))
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state_key, action)] = new_q

    def get_possible_actions(self, state_key):
        """Convert state key back to board to find empty cells"""
        board = np.array(list(state_key)).reshape(BOARD_SIZE, BOARD_SIZE)
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) 
                if board[r][c] == ' ']

class GameEngine:
    """Manages game state and coordinates AI agents"""
    
    def __init__(self):
        """Initialize game state and load/create AI agents"""
        # Initialize empty game board
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), ' ', dtype='U1')
        self.players = ['X', 'O']          # Player identifiers
        self.current_player = 0            # Current player index
        self.game_over = False             # Game status flag
        self.winner = None                 # Game outcome
        self.move_history = []             # Record of all moves
        self.x_wins = 0                    # Win counter for X
        self.o_wins = 0                    # Win counter for O
        self.draws = 0                     # Draw counter

        # Load existing agents or create new ones
        if os.path.exists(Q_FILE):
            with open(Q_FILE, 'rb') as f:
                self.agents = pickle.load(f)
                # Backward compatibility for loaded agents
                for agent in self.agents.values():
                    if not hasattr(agent, 'last_state'):
                        agent.last_state = None
                        agent.last_action = None
        else:
            # Create new Q-learning agents
            self.agents = {
                'X': QLearningAgent('X'),
                'O': QLearningAgent('O')
            }

    def save_progress(self):
        """Save current agent states to file"""
        with open(Q_FILE, 'wb') as f:
            pickle.dump(self.agents, f)

    def reset(self):
        """Reset game state for new match"""
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), ' ', dtype='U1')
        self.current_player = random.randint(0, 1)  # Random starting player
        self.game_over = False
        self.winner = None
        self.move_history = []

    def check_winner(self, player):
        """Check for winning condition in all directions"""
        directions = [(1,0), (0,1), (1,1), (1,-1)]  # Horizontal, vertical, diagonals
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] != player:
                    continue  # Skip non-player cells
                
                # Check all four possible winning directions
                for dx, dy in directions:
                    count = 1  # Start count with current cell
                    for i in range(1, WIN_CONDITION):
                        r = row + dx * i
                        c = col + dy * i
                        # Validate boundaries and player match
                        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == player:
                            count += 1
                        else:
                            break  # Stop checking this direction
                    if count >= WIN_CONDITION:
                        return True  # Winning line found
        return False  # No winner

    def play_step(self):
        """Execute one game step (move by current player)"""
        if self.game_over:
            return None  # Game already concluded

        player = self.players[self.current_player]
        agent = self.agents[player]
        state_key = agent.get_state_key(self.board)
        available_actions = agent.get_possible_actions(state_key)
        
        if not available_actions:  # No moves remaining
            self.game_over = True
            self.draws += 1
            return None

        # Select action using agent's policy
        action = agent.choose_action(state_key, available_actions)
        if action is None:  # No valid action selected
            return None

        # Track opponent's previous state for penalty
        opponent = 'O' if player == 'X' else 'X'
        opponent_agent = self.agents[opponent]
        prev_opponent_state = opponent_agent.last_state
        prev_opponent_action = opponent_agent.last_action

        # Record current agent's state/action
        agent.last_state = state_key
        agent.last_action = action

        # Execute the selected move
        row, col = action
        self.board[row, col] = player
        next_state_key = agent.get_state_key(self.board)
        
        # Store move history for visualization
        self.move_history.append({
            'player': player,
            'position': (int(row), int(col)),
            'board': self.board.tolist()  # Convert numpy array to list
        })

        # Check for win condition
        if self.check_winner(player):
            self.game_over = True
            self.winner = player
            reward = 10  # High reward for winning
            
            # Apply penalty to opponent's last move
            if prev_opponent_state and prev_opponent_action:
                opponent_agent.update_q_value(
                    prev_opponent_state,
                    prev_opponent_action,
                    -5,  # Negative reward for losing
                    next_state_key
                )
            
            # Update win counters
            if player == 'X':
                self.x_wins += 1
            else:
                self.o_wins += 1
        else:
            reward = 0.1  # Small reward for valid move

        # Update Q-value for current agent
        agent.update_q_value(state_key, action, reward, next_state_key)
        
        # Switch to next player
        self.current_player = 1 - self.current_player
        return self.move_history[-1]

# Flask Routes ----------------------------------------------------------------

@app.route('/')
def index():
    """Serve main interface with training controls"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """Handle training requests and run specified number of games"""
    games = int(request.form.get('games', 10))
    game_engine = GameEngine()
    
    # Train through specified number of games
    for _ in range(games):
        game_engine.reset()
        while not game_engine.game_over:
            game_engine.play_step()
        game_engine.save_progress()
    
    return jsonify({
        'status': 'success',
        'message': f'Trained {games} games',
        'q_table_size': sum(len(agent.q_table) for agent in game_engine.agents.values()),
        'stats': {
            'x_wins': game_engine.x_wins,
            'o_wins': game_engine.o_wins,
            'draws': game_engine.draws
        }
    })

@app.route('/play')
def play_game():
    """Demonstrate a full game between trained agents"""
    game_engine = GameEngine()
    game_engine.reset()
    
    # Disable exploration during demonstration
    game_engine.agents['X'].epsilon = 0
    game_engine.agents['O'].epsilon = 0
    
    # Play through complete game
    while not game_engine.game_over:
        game_engine.play_step()
    
    return render_template('game.html', 
                         moves=game_engine.move_history,
                         winner=game_engine.winner)

@app.route('/stats')
def get_stats():
    """Return current training statistics"""
    game_engine = GameEngine()
    return jsonify({
        'x_wins': game_engine.x_wins,
        'o_wins': game_engine.o_wins,
        'draws': game_engine.draws
    })

if __name__ == '__main__':
    # Start Flask development server
    app.run(debug=True)