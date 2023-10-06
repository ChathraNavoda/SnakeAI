import random
from turtle import done
import numpy as np
from collections import deque

import torch
from game import SnakeGameAI, Direction, Point
from double_dqn_model import DoubleLinear_QNet, DoubleQTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.95

class DoubleDQNAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # parameters to control the randomness
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = DoubleLinear_QNet(11, 256, 3)
        self.target_model = DoubleLinear_QNet(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model with model weights
        self.trainer = DoubleQTrainer(self.model, self.target_model, lr=LR, gamma=GAMMA)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DoubleDQNAgent()
    game = SnakeGameAI()
    
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Remember the state
        agent.remember(state_old, final_move, reward, state_new, done)

        # Train agent using experiences from memory
        if done:
            if len(agent.memory) >= BATCH_SIZE:
                mini_batch = random.sample(agent.memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*mini_batch)
                agent.trainer.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
            
            # Update target network every 1000 steps
            if agent.n_games % 5 == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

            # Reset the game, train long memory, and plot results
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
