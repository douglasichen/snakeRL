from math import exp
import torch
import random
import numpy as np
from enum import Enum
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001



class Agent:

	def __init__(self):
		self.n_games = 0
		self.epsilon = 0 # randomness
		self.gamma = 0.9 # discount rate
		self.memory = deque(maxlen=MAX_MEMORY) # popleft()
		self.model = Linear_QNet(11, 256, 3)
		self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

	def raycast_distance(self, game, pos, dir):
		# pos = Point(pos.x+dir.x, pos.y+dir.y)
		# dist = 0.1
		
		# while (not game.is_collision(pos)):
		# 	dist += 1
		# 	pos = Point(pos.x+dir.x, pos.y+dir.y)
		
		# return 1.0/dist
		return game.is_collision(Point(pos.x+dir.x, pos.y+dir.y))


	def get_state(self, game):
		head = game.snake[0]
		
		vector_dir = [ Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1) ] # clockwise
		dir_ind = int(game.direction)
		state = [
			# straight danger distance
			self.raycast_distance(game, head, vector_dir[dir_ind]),

			# right danger distance
			self.raycast_distance(game, head, vector_dir[(dir_ind + 1)%4]),

			# left danger distance
			self.raycast_distance(game, head, vector_dir[(dir_ind - 1)%4]),
			
			
			# Move direction
			game.direction == Direction.RIGHT,
			game.direction == Direction.DOWN,
			game.direction == Direction.LEFT,
			game.direction == Direction.UP,
			
			# Food location 
			game.food.x < game.head.x,  # food left
			game.food.x > game.head.x,  # food right
			game.food.y < game.head.y,  # food up
			game.food.y > game.head.y  # food down
		]

		# state = []
		# for y in range(game.grid_height):
		# 	for x in range(game.grid_width):
		# 		state.append(game.grid[y][x])

		return np.array(state, dtype=int)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

	def train_long_memory(self):
		if len(self.memory) > BATCH_SIZE:
			mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
		else:
			mini_sample = self.memory

		states, actions, rewards, next_states, dones = zip(*mini_sample)
		self.trainer.train_step(states, actions, rewards, next_states, dones, long_term=True)
		


		#for state, action, reward, nexrt_state, done in mini_sample:
		#    self.trainer.train_step(state, action, reward, next_state, done)

	def train_short_memory(self, state, action, reward, next_state, done):
		self.trainer.train_step(state, action, reward, next_state, done)

	def get_action(self, state):
		# random moves: tradeoff exploration / exploitation
		# a = -0.98
		# k = 0.1
		# d = 50
		# c = 1
		# self.epsilon = a / (1 + exp(-k*(self.n_games-d))) + c

		# self.epsilon = -0.02*self.n_games+1 if self.n_games<=45 else 0.01

		# self.epsilon = max(-0.005*self.n_games+0.4, 0.001)

		# self.epsilon = max(-0.0015*self.n_games+0.3, 0.001)

		# self.epsilon = max(-0.004*self.n_games+0.7, 0.001)

		self.epsilon = max(-0.005*self.n_games + 0.4, 0.0001)
		# if (self.n_games>120):
		# 	self.epsilon = 0
		
		final_move = [0,0,0]
		if random.random() <= self.epsilon:
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
	agent = Agent()
	game = SnakeGameAI()
	while True:
		# get old state
		state_old = agent.get_state(game)

		# get move
		final_move = agent.get_action(state_old)

		# perform move and get new state
		reward, done, score = game.play_step(final_move)
		state_new = agent.get_state(game)

		# train short memory
		agent.train_short_memory(state_old, final_move, reward, state_new, done)

		# remember
		agent.remember(state_old, final_move, reward, state_new, done)

		if done:
			# train long memory, plot result
			game.reset()
			agent.n_games += 1
			agent.train_long_memory()

			if score > record:
				record = score
				agent.model.save()

			# print('Game', agent.n_games, 'Score', score, 'Record:', record)

			plot_scores.append(score)
			total_score += score
			mean_score = total_score / agent.n_games
			plot_mean_scores.append(mean_score)
			plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
	train()