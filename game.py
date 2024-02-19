import pygame
import random
from enum import IntEnum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(IntEnum):
	RIGHT = 0
	DOWN = 1
	LEFT = 2
	UP = 3

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 4000

class SnakeGameAI:

	def __init__(self, screen_width=640, screen_height=480):
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.grid_width = screen_width//BLOCK_SIZE
		self.grid_height = screen_height//BLOCK_SIZE

		# init display
		self.display = pygame.display.set_mode((self.screen_width, self.screen_height))
		pygame.display.set_caption('Snake')
		self.clock = pygame.time.Clock()
		self.reset()

	def reset(self):
		# init game state
		self.direction = Direction.RIGHT

		self.head = Point(self.grid_width//2, self.grid_height//2)
		self.snake = [self.head,
					  Point(self.head.x-1, self.head.y),
					  Point(self.head.x-2, self.head.y)]
		
		self.grid = [ [0]*(self.grid_width) for _ in range(self.grid_height)]
		for p in self.snake:
			self.grid[p.y][p.x] = 1

		self.score = 0
		self.food = None
		self._place_food()
		self.frame_iteration = 0


	def _place_food(self):
		# x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
		# y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
		# self.food = Point(x, y)
		# if self.food in self.snake:
		# 	self._place_food()
		n = random.randint(0, self.grid_width*self.grid_height - len(self.snake) - 1)
		for y in range(self.grid_height):
			for x in range(self.grid_width):
				if (self.grid[y][x] == 0):
					if (n == 0):
						self.food = Point(x,y)
						self.grid[y][x] = 2
						return
					n -= 1


	def play_step(self, action):
		self.frame_iteration += 1
		# 1. collect user input
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		
		# 2. move
		self._move(action) # update the head

		
		# 3. check if game over
		reward = 0
		game_over = False
		if self.is_collision() or self.frame_iteration > 100*len(self.snake):
			game_over = True
			reward = -10
			return reward, game_over, self.score

		# update grid AFTER collision check
		self.grid[self.head.y][self.head.x] = 1

		# 4. place new food or just move
		if self.head == self.food:
			self.score += 1
			reward = 10
			self._place_food()
		else:
			self.grid[self.snake[-1].y][self.snake[-1].x] = 0
			self.snake.pop()
			# reward = -1
		
		# 5. update ui and clock
		self._update_ui()
		self.clock.tick(SPEED)
		# 6. return game over and score
		return reward, game_over, self.score


	def is_collision(self, p=None):
		if p is None:
			p = self.head
		# hits boundary
		if p.x<0 or p.x>=self.grid_width or p.y<0 or p.y>=self.grid_height:
			return True

		# hits itself
		if self.grid[p.y][p.x] == 1:
			return True

		return False


	def _update_ui(self):
		self.display.fill(BLACK)

		for pt in self.snake:
			pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
			pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x*BLOCK_SIZE+4, pt.y*BLOCK_SIZE+4, 12, 12))

		pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x*BLOCK_SIZE, self.food.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

		text = font.render("Score: " + str(self.score), True, WHITE)
		self.display.blit(text, [0, 0])
		pygame.display.flip()


	def _move(self, action):
		# [straight, right, left]

		clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		idx = clock_wise.index(self.direction)

		if np.array_equal(action, [1, 0, 0]):
			new_dir = clock_wise[idx] # no change
		elif np.array_equal(action, [0, 1, 0]):
			next_idx = (idx + 1) % 4
			new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
		else: # [0, 0, 1]
			next_idx = (idx - 1) % 4
			new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

		self.direction = new_dir

		x = self.head.x
		y = self.head.y
		if self.direction == Direction.RIGHT:
			x += 1
		elif self.direction == Direction.LEFT:
			x -= 1
		elif self.direction == Direction.DOWN:
			y += 1
		elif self.direction == Direction.UP:
			y -= 1

		self.head = Point(x, y)
		
		self.snake.insert(0, self.head)