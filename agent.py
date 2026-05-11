import torch
import random
import numpy as np
from collections import deque
from snake_game_env import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer  # ### 1. IMPORT THE BRAIN
# from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        # ### 2. INITIALIZE BRAIN AND TEACHER
        # Input: 11 states, Hidden: 256, Output: 3 actions
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.trainer.load_checkpoint()
    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x , head.y - BLOCK_SIZE)
        point_d = Point(head.x , head.y + BLOCK_SIZE)

        point_l_2 = Point(head.x - ( 2 * BLOCK_SIZE), head.y )
        point_r_2 = Point(head.x + ( 2 * BLOCK_SIZE), head.y )
        point_u_2 = Point(head.x, head.y  - ( 2 * BLOCK_SIZE) )
        point_d_2 = Point(head.x, head.y  + ( 2 * BLOCK_SIZE) )

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 1. Danger Straight?
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # 2. Danger Right?
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # 3. Danger Left?
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # 4. Danger 2 Straight?
            (dir_r and game.is_collision(point_r_2))or
            (dir_l and game.is_collision(point_l_2))or
            (dir_u and game.is_collision(point_u_2))or
            (dir_d and game.is_collision(point_d_2)),

            # 5. Danger 2 Right?
            (dir_u and game.is_collision(point_r_2))or
            (dir_d and game.is_collision(point_l_2))or
            (dir_l and game.is_collision(point_u_2))or
            (dir_r and game.is_collision(point_d_2)),

            # 6. Danger 2 Left?
            (dir_d and game.is_collision(point_r_2))or
            (dir_u and game.is_collision(point_l_2))or
            (dir_r and game.is_collision(point_u_2))or
            (dir_l and game.is_collision(point_d_2)),


            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location 
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y   # Food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Randomly pick 1000 memories
        else:
            mini_sample = self.memory

        # Unpack the memories
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # ### 3. TRAIN ON BATCH
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # ### 4. TRAIN ON SINGLE STEP
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games # Randomness decreases as games increase
        self.epsilon = 150 - self.n_games
        final_move = [0,0,0]
        
        if self.epsilon < 0:
            self.epsilon = 0

        # If random number is small, explore (Random Move)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        
        # Else, exploit (Ask the Brain)
        else:
            # Convert state to tensor for PyTorch
            state0 = torch.tensor(state, dtype=torch.float)
            
            # ### 5. GET PREDICTION FROM MODEL
            prediction = self.model(state0) # Pass state to brain
            move = torch.argmax(prediction).item() # Pick the highest number
            final_move[move] = 1

        return final_move

    def save_agent_state(self):
        return { 
            'n_games': self.n_games
        }

    def load_agent_state(self, state):
        self.n_games = state.get('n_games', 0)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("Starting Training...")

    if os.path.exists('./model/agent_state.pth'):
        data = torch.load('./model/agent_state.pth')
        agent.load_agent_state(data['agent_state'])
        record = data.get('record', 0)
        print("♻️ Agent state restored")

    
    while True:
        # 1. Get old state
        state_old = agent.get_state(game)

        # 2. Get move
        final_move = agent.get_action(state_old)

        # 3. Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 6. Train long memory (Replay Experience)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                
                # ### 6. SAVE THE MODEL
                agent.model.save()
                agent.trainer.save_checkpoint()

                torch.save({
                    'agent_state': agent.save_agent_state(),
                    'record': record
                }, './model/agent_state.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            
            # # 2. Draw the graph
            # plot(plot_scores, plot_mean_scores)
if __name__ == '__main__':
    train()