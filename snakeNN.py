import random
import sys
import time

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


import pygame
from pygame.locals import *

from game import snakeGame
from game import replay

from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
from tensorflow import keras
import os
import math



pygame.init()

fps = 120
fpsClock = pygame.time.Clock()

width, height = 800, 800
gameDisplay = pygame.display.set_mode((width, height))

class SnakeNN():

    def __init__(self, initialGames=200, testGames = 0, goalSteps = 200, lr=1e-2, filename = 'nn_model.tflearn'):
        self.initial_games = initialGames
        self.test_games = testGames
        self.goal_steps = goalSteps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]

    def initial_population(self, firstTrain=True):
        """Trains the initial population"""
        training_data = []

        if not firstTrain:
            
            tf.reset_default_graph()
            model = self.model()
            model.load(os.getcwd()+"/"+self.filename, weights_only=True)

        for _ in range(self.initial_games):
            game = snakeGame()
            _, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            food_distance = 999

            run = True
            
            while run:


      
                if firstTrain or random.randint(0,100) <= 25:
                    action, game_action = self.generate_action(snake)
                else:
                    predictions = []
                    for action in range(-1, 2):
                       predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                    action = np.argmax(np.array(predictions))
                    game_action = self.get_game_action(snake, action - 1)

                    for i in range(10):
                        if self.doesThisKillYourself(snake, game_action):
                            game_action = self.get_game_action(snake, random.randint(0, 2)-1)

                
         
                

                            
                done, score, snake, food = game.run(game_action)

                if not done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), score])
                    run = False

                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    training_data.append([self.add_action_to_observation(prev_observation, action), score])
       
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
                    prev_score = score

                

        return training_data

    def doesThisKillYourself(self, snake, game_action):
        """Used for making sure the snake doesn't do a double back and destroy itself"""
        if len(snake) != 1:
            for key in self.vectors_and_keys:
                if (key[1]+2)%4 == game_action:
                    return self.is_direction_blocked([snake[-1][0]+key[0][0], snake[-1][1]+key[0][1]], self.get_snake_direction_vector(snake))
                    

        return False

    def generate_action(self, snake):
        action = random.randint(0,2) - 1
        return action, self.get_game_action(snake, action)
        

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction

        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        if action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        

        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                gameAction = pair[1]

        
        return gameAction

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        if len(snake) == 1:
            return np.array([0, -1])
        return np.array(snake[-1])-np.array(snake[-2])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[-1])

    def normalize_vector(self, vector):
        if np.linalg.norm(vector) == 0:
            return np.array([0, 0])
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[-1]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] <= 0 or point[1] <= 0 or point[0] == 21 or point[1] == 21

    
    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)

        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        tf.reset_default_graph()
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename,snapshot_epoch=True, snapshot_step=100)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        counter = 0
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = snakeGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food  = game.run(game_action)
                game_memory.append([prev_observation, action])
                if not done:
                    print('-----')
                    print(steps)
                    print(snake)
                    print(food)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)

            counter += 1

            if counter % int(self.test_games/10) == 0:
                print("Testing... Step " + str(counter) + "/" + str(self.test_games))
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    

    def visualise_game(self, model):
        game = snakeGame(gui = True)
        width, height = game.width, game.height
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        step = 0
        stop = False
        while not stop:

            food_distance = self.get_food_distance(snake, food)



            precictions = []
            for action in range(-1, 2):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)

            for i in range(10):
                if self.doesThisKillYourself(snake, game_action):
                    game_action = self.get_game_action(snake, random.randint(0, 2)-1)


            
                          
            done, _, snake, food  = game.run(game_action)
            if not done:
                stop = True
                print(self.generate_observation(snake, food), game_action, self.get_snake_direction_vector(snake), snake[-1])
                game.run(9)
                break
            else:
                prev_observation = self.generate_observation(snake, food)
            #step += 1

            #if step % 1000 == 0 and step != 0:
            #    stop = True
            #    break

        

    def train(self, trainingData, model=False):
        if model == False:
            nn_model = self.model()
            nn_model = self.train_model(trainingData, nn_model)
        else:
            nn_model = self.train_model(trainingData, model)
        return nn_model
        

    def visualise(self, model=False):
        tf.reset_default_graph()
        if model == False:
            nn_model = self.model()
            nn_model.load(os.getcwd()+"/"+self.filename, weights_only=True)
            self.visualise_game(nn_model)
        else:
            self.visualise_game(model)

    def test(self):
        nn_model = self.model()
        nn_model.load("./"+self.filename, weights_only=True)
        self.test_model(nn_model)
        

if __name__ == "__main__":
  
    snakeNN = SnakeNN()
    #snakeNN.visualise()

    
    #Loading old model
    #model = snakeNN.model()
    #model.load(os.getcwd()+"/"+snakeNN.filename, weights_only=True)

    #New Model
    model = snakeNN.train(snakeNN.initial_population())
    snakeNN.visualise(model)
    while True:
       model = snakeNN.train(snakeNN.initial_population(False), model)
       snakeNN.visualise(model)
   


    

