from math import exp
import torch
import random
import numpy as np
from enum import Enum
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
DIRS = [ Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1) ] # clockwise


class Agent:

	def __init__(self, file_name=None):
		self.n_games = 0
		self.epsilon = 0 # randomness
		self.gamma = 0.9 # discount rate
		self.memory = deque(maxlen=MAX_MEMORY) # popleft()
		self.input_size = 14
		self.model = Linear_QNet(self.input_size, 256, 3)
		self.loaded = False
		if file_name!=None:
			self.model.load(file_name)
			self.loaded = True
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

		dir_ind = int(game.direction)
		straight_area = self.get_area(game, add_points(game.head, DIRS[dir_ind]))
		right_area = self.get_area(game, add_points(game.head, DIRS[(dir_ind+1)%4]))
		left_area = self.get_area(game, add_points(game.head, DIRS[(dir_ind-1)%4]))
		area_sum = straight_area + right_area + left_area

		straight_danger = self.raycast_distance(game, head, DIRS[dir_ind])
		right_danger = self.raycast_distance(game, head, DIRS[(dir_ind + 1)%4])
		left_danger = self.raycast_distance(game, head, DIRS[(dir_ind - 1)%4])
		danger_sum = straight_danger + right_danger + left_danger

		
		state = [
			# straight danger distance
			(0 if danger_sum==0 else straight_danger/danger_sum),

			# right danger distance
			(0 if danger_sum==0 else right_danger/danger_sum),

			# left danger distance
			(0 if danger_sum==0 else left_danger/danger_sum),
			
			# straight area
			(0 if area_sum==0 else straight_area/area_sum),

			# right area
			(0 if area_sum==0 else right_area/area_sum),

			# left area
			(0 if area_sum==0 else left_area/area_sum),

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
		# state.append(game.head.y)
		# state.append(game.head.x)
		# for y in range(game.grid_height):
		# 	for x in range(game.grid_width):
		# 		state.append(game.grid[y][x])
		# print(state)
		return np.array(state, dtype=float)


	def get_area(self, game, at):
		if game.out_of_bounds(at) or game.grid[at.y][at.x]==1:
			return 0

		# make copy of game grid
		vis = [ [0]*(game.grid_width) for _ in range(game.grid_height)]
		for y in range(game.grid_height):
			for x in range(game.grid_width):
				vis[y][x] = game.grid[y][x]

		# dfs count
		stack = [at]
		ans = 0
		vis[at.y][at.x] = 1
		while len(stack) > 0:
			point = stack.pop()
			ans += 1
			for dir in DIRS:
				new_point = add_points(point, dir)
				if not game.out_of_bounds(new_point) and vis[new_point.y][new_point.x] != 1:
					stack.append(new_point)
					vis[new_point.y][new_point.x] = 1
		return ans

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

		# self.epsilon = max(-0.005*self.n_games + 0.4, 0.0001)
		# self.epsilon = max(-0.002*self.n_games + 0.6, 0.0001)

		a = 64
		b = -80
		c = -0.18
		# a = 64
		# b = -65
		# c = -0.3
		# a = 64
		# b = -65
		# c = -0.4
		f = a/(self.n_games-b)+c
		self.epsilon = max(f, 0.00001)
		# print(self.epsilon)
		# if (self.n_games>120):
		# 	self.epsilon = 0
		
		# if self.loaded:
		# 	self.epsilon = 0.01
		final_move = [0,0,0]
		if random.random() < self.epsilon:
			move = random.randint(0,2)
		if self.is_about_to_die(state):
			# move = random.randint(1, 2)
			if state[4] > state[5]:
				move = 1
			elif state[5] > state[4]:
				move = 2
			elif state[3]!=state[4]:
				move = 0
			else:
				move = random.randint(1,2)
			final_move[move] = 1
			# print("about to die chose", move)
		else:
			state0 = torch.tensor(state, dtype=torch.float)
			prediction = self.model(state0)
			move = torch.argmax(prediction).item()
			final_move[move] = 1

		return final_move
	
	def is_about_to_die(self, state):
		return state[4] != state[5] and state[3]==0


def train(file_name=None):
	plot_scores = []
	plot_mean_scores = []
	total_score = 0
	record = 0
	agent = Agent(file_name)
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
			
			# save
			if agent.n_games%100 == 0:
				agent.model.save('model_fixed_int2float_bug.pth')

def run(file_name):
	plot_scores = []
	plot_mean_scores = []
	total_score = 0
	agent = Agent(file_name)
	game = SnakeGameAI()
	while True:
		# get old state
		state_old = agent.get_state(game)

		# get move
		final_move = agent.get_action(state_old)

		# perform move and get new state
		reward, done, score = game.play_step(final_move)
		
		state_new = agent.get_state(game)
		if reward != 'went':
			print(state_new)


		# # train short memory
		# agent.train_short_memory(state_old, final_move, reward, state_new, done)

		# # remember
		# agent.remember(state_old, final_move, reward, state_new, done)

		if done:
			# print(state_old)
			# # print(state_new)
			# break
			# train long memory, plot result
			game.reset()
			agent.n_games += 1
			# agent.train_long_memory()

			plot_scores.append(score)
			total_score += score
			mean_score = total_score / agent.n_games
			plot_mean_scores.append(mean_score)
			plot(plot_scores, plot_mean_scores)

			# save
			if agent.n_games%50 == 0:
				agent.model.save(file_name='model_fixed_1.pth')


if __name__ == '__main__':
	# train(file_name='model3.pth')
	run('model_fixed_1.pth')
	# train()