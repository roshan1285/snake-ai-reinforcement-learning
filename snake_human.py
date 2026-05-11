import pygame
import random
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3 
    DOWN = 4
# 1. Initialize
pygame.init()

# 2. Constants (Settings)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 153, 213)
RED = (255,0,0)
GREEN = (0,255,0)

WIDTH = 600
HEIGHT = 400

BLOCK_SIZE = 10
SNAKE_SPEED = 10

snake_List = []
Length_of_snake = 1

clock = pygame.time.Clock()
x1_change = 0
y1_change = 0

screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Snake Game by R")


game_over = False
x1 = WIDTH / 2  # Start in middle (300)
y1 = HEIGHT / 2 # Start in middle (200)

foodx = round(random.randrange(0, WIDTH - 10) / 10.0) * 10.0
foody = round(random.randrange(0, HEIGHT - 10) / 10.0) * 10.0

direction = None

def gameLoop():
    global game_over, x1, y1, x1_change, y1_change, snake_List, Length_of_snake, clock, foodx, foody, direction
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if direction != Direction.RIGHT:
                        x1_change = -10
                        y1_change = 0
                        direction = Direction.LEFT

                if event.key == pygame.K_RIGHT:
                    if direction != Direction.LEFT:
                        x1_change = 10
                        y1_change = 0
                        direction = Direction.RIGHT
                if event.key == pygame.K_UP:
                    if direction != Direction.DOWN:
                        y1_change = -10
                        x1_change = 0
                        direction = Direction.UP
                    
                if event.key == pygame.K_DOWN:
                    if direction != Direction.UP:
                        y1_change = 10
                        x1_change = 0
                        direction = Direction.DOWN
            
        

        x1 += x1_change
        y1 += y1_change

       

        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)

        if len(snake_List) > Length_of_snake:
            del snake_List[0]
        
        if x1 >= WIDTH or x1 < 0 or y1>=HEIGHT or y1 < 0:
        #     game_over = True
            if x1 >= WIDTH:
                x1 = 0
            elif x1 < 0:
                x1 = WIDTH
            elif y1 >= HEIGHT:
                y1 = 0
            elif y1 < 0:
                y1 = HEIGHT

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_over = True

        screen.fill(BLACK)
        
        for x in snake_List:
            # print(x)
            pygame.draw.rect(screen, BLUE,[x[0],x[1],BLOCK_SIZE,BLOCK_SIZE])

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, WIDTH - 10) / 10.0 ) * 10.0
            foody = round(random.randrange(0, HEIGHT - 10) / 10.0 ) * 10.0 

            Length_of_snake += 1
        
        pygame.draw.rect(screen,BLUE,[x1,y1,BLOCK_SIZE,BLOCK_SIZE])
        pygame.draw.rect(screen,RED,[foodx, foody, BLOCK_SIZE, BLOCK_SIZE])

        pygame.display.update()

        clock.tick(15)

    pygame.quit()

gameLoop()