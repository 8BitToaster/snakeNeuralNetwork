import random
import sys
import time


import pygame
from pygame.locals import *

pygame.init()

fps = 120
fpsClock = pygame.time.Clock()

width, height = 800, 800
gameDisplay = pygame.display.set_mode((width, height))


class snakeGame():

    def __init__(self, boardWidth=20, boardHeight=20, gui=False):
        self.width, self.height = boardWidth, boardHeight
        self.gui = gui

        self.board = []
        for j in range(self.height):
            temp = []
            for i in range(self.width):
                temp.append(0)
            self.board.append(temp)

        self.score = 0 #Scoring for evaulation

        self.foodPos = [] #Stores the food data

        #Storing Snakes Data
        self.snakePos = [int(self.width/2), int(self.height/2)]
        self.snake = [[self.snakePos[0], self.snakePos[1]]]
        self.snakeLength = 5

        self.sinceLastFood = 0 #kills people if it takes forever to get food

        #Records for replays
        self.recordedSnake = []
        self.recordedFood = []
        self.lastDirection = 0
        self.direction = 0 #0: up, 1: down, 2:left, 3:right
        self.direct = []
        self.appleScore = 0

        self.key = 0 #Running a simulation from another file

        self.game_run = True
        
    def start(self):
        self.generateFood()

        return self.generateObservation()


    def generateFood(self):
        """Generates a food on the board"""
        stop = False
        while not stop:
            ranX = random.randint(1, self.width-2)
            ranY = random.randint(1, self.height-2)

            if self.board[ranY][ranX] == 0 and [ranX, ranY] not in self.snake:
                self.board[ranY][ranX] = 2
                stop = True
                self.sinceLastFood = 0

            self.foodPos = [ranX, ranY]


    def run(self, step, record=False):
        """Main loop for a run"""

        self.score = -1
        pos = [self.snakePos[0], self.snakePos[1]]

        self.recordedFood.append([self.foodPos[0], self.foodPos[1]])
        temp = []
        for item in self.snake:
            temp.append(item)
        self.recordedSnake.append(temp)
        
        if self.gui:
            self.draw()

        if len(self.snake) >= self.snakeLength:
            self.snake.pop(0)

        self.movement(step, record)

        if self.collided() == 1: #Collided with itself
            self.endGame()
  
        if self.collided() == 2: #Collisdes with Food
            self.snakeLength += 1
            self.score = 2
            self.appleScore += 1
            self.generateFood()


        #If you take forever to find food
        self.sinceLastFood += 1

        if self.sinceLastFood == 300:
            self.endGame()

   
        if abs(self.snakePos[0]-self.foodPos[0])+abs(self.snakePos[1]-self.foodPos[1]) < abs(pos[0]-self.foodPos[0])+abs(pos[1]-self.foodPos[1]):
            self.score = 1
        

        return self.generateObservation()
            

    def draw(self):
        """Draws everything, if gui == True"""

        gameDisplay.fill((200, 200, 200))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        for i, part in enumerate(self.snake):
            if i != len(self.snake)-1:
                pygame.draw.rect(gameDisplay, (255, 0, 0), (int(800/self.width*part[0]), int(800/self.height*part[1]), int(800/self.width), int(800/self.height)), 0)
            else:
                pygame.draw.rect(gameDisplay, (155, 0, 0), (int(800/self.width*part[0]), int(800/self.height*part[1]), int(800/self.width), int(800/self.height)), 0)

        pygame.draw.rect(gameDisplay, (0, 250, 0), (int(800/self.width*self.foodPos[0]), int(800/self.height*self.foodPos[1]), int(800/self.width), int(800/self.height)), 0)

        pygame.display.flip()
        fpsClock.tick(120)

    def movement(self, direction, record):
        #0: left, 1: down, 2: right, 3: up


        if direction == 0:
            self.snakePos[0] -= 1

        if direction == 1:
            self.snakePos[1] += 1

        if direction == 2:
            self.snakePos[0] += 1

        if direction == 3:
            self.snakePos[1] -= 1

        #Old Rotation Method
        """
        if direction == 0: #Forward
            for item in [[0, [0, -1]], [1, [0, 1]], [2, [-1, 0]], [3, [1, 0]]]:
                if self.direction == item[0]:
                    self.snakePos[1] += item[1][1]
                    self.snakePos[0] += item[1][0]
                    
        elif direction == 1: #Left
            stop = False
            for item in [[0, [-1, 0], 2], [1, [1, 0], 3], [2, [0, 1], 1], [3, [0, -1], 0]]:
                if self.direction == item[0] and not stop:
                    self.snakePos[1] += item[1][1]
                    self.snakePos[0] += item[1][0]

                    stop = True
                    self.direction = item[2]
                         
        elif direction == 2: #Right
            stop = False
            for item in [[0, [1, 0], 3], [1, [-1, 0], 2], [2, [0, -1], 0], [3, [0, 1], 1]]:
                if self.direction == item[0] and not stop:
                    self.snakePos[1] += item[1][1]
                    self.snakePos[0] += item[1][0]
                    self.direction = item[2]

                    stop = True
        """

        self.lastDirection = direction

        self.snake.append([self.snakePos[0], self.snakePos[1]])

        #Returns data to Neural Network File
        if __name__ != "__main__":
            return self.generateObservation()
        

    def collided(self):
        for i, part in enumerate(self.snake):
            if i != len(self.snake)-1 and part == self.snakePos:
                return 1

        if self.snakePos[0] in [-1, len(self.board[0])] or self.snakePos[1] in [-1, len(self.board)]:
            return 1

        if self.snakePos == self.foodPos:
            return 2

        return 0

    def generateObservation(self):
        return self.game_run, self.score, self.snake, self.foodPos

    def endGame(self):
        """Instantly ends the game, no questions asked"""
        self.game_run = False

    def restart(self):
        """If you don't like it, try it again! -Probably someone"""

        self.board = []
        for j in range(self.height):
            temp = []
            for i in range(self.width):
                temp.append(0)
            self.board.append(temp)

        self.score = 0 #Scoring for evaulation

        self.foodPos = [] #Stores the food data

        #Storing Snakes Data
        self.snakePos = [int(self.width/2), int(self.height/2)]
        self.snake = [[self.snakePos[0], self.snakePos[1]]]
        self.snakeLength = 5

def replay(recordedSnake, recordedFood, recordScore, width=20, height=20):
    counter = 0

    while counter != len(recordedSnake):

        time.sleep(1)

        gameDisplay.fill((200, 200, 200))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


        for i, part in enumerate(recordedSnake[counter]):
            if i != len(recordedSnake[counter])-1:
                pygame.draw.rect(gameDisplay, (255, 0, 0), (int(800/width*part[0]), int(800/height*part[1]), int(800/width), int(800/height)), 0)
            else:
                pygame.draw.rect(gameDisplay, (150, 0, 0), (int(800/width*part[0]), int(800/height*part[1]), int(800/width), int(800/height)), 0)

        pygame.draw.rect(gameDisplay, (0, 250, 0), (int(800/width*recordedFood[counter][0]), int(800/height*recordedFood[counter][1]), int(800/width), int(800/height)), 0)

        pygame.display.flip()
        fpsClock.tick(120)
        
        counter += 1

    
        
        
    
                
if __name__ == "__main__":
    score = 0
    Max = 0
    Replays = []
    for i in range(10000):
        snake = snakeGame(20, 20, False)

        snake.start()
        running = True

        step = 0
        while running:
            score = snake.run(random.randint(0,3))
            running = score[0]
            step += 1
            
        

        if step > Max:
            Max = step
            Replays = [snake.recordedSnake, snake.recordedFood]

    print(Max)
