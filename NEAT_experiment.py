from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing
import tensorflow as tf
import itertools as it
import numpy as np
import neat
import time
import gym
import os


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    gray = gray / 255
    return gray


def collision(obs, frame):
    if frame < 12: 
        return False
    
    car = 0
    for y in range(66, 77):
        for x in range(45,51):
            car += round(obs[y][x])
            # print('{:.0f} '.format(obs[y][x]), end='')
        # print()
    return False if car <= 0 else True


def test():
    print('Start Game!')
    env = gym.make('CarRacing-v1')
    print(env)
    
    action = [0,0,0]
    env.reset()
    # env.car = Car(env.world, *env.track[0][1:4])
    total_reward = 0.0
    steps = 0
    
    print(env.car)
    # env.car = Car(env.world, *env.track[0][1:4])
    print(env.car)

    stop = False
    while not stop:
        print(env.car)
        obs, r, _, _ = env.step(action)
        obs = rgb2gray(obs)
        
        total_reward += r
        if collision(obs, steps):
            print("Done! Reward -> {}".format(total_reward))
            stop = True
        steps += 1
        env.render()


def eval_genomes(genomes, config):
    print('evaling')
    env = CarRacing()
    

    left_right = [-1, 0, 1]
    acceleration = [1, 0]
    brake = [0.2, 0]
    actions = np.array([action for action in it.product(left_right, acceleration, brake)])

    scores = []

    nets = []
    cars = []
    g = []
    for id, genome in genomes:
        score = 0
        genome.fitness = 0
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        track = env._create_track()
        cars.append(Car(env.world, *track[0][1:4]))
        g.append(genome)


    obs = env.reset()
    obs = rgb2gray(obs)

    for i in range(1000) and len(cars):

        for car in cars:
            # determine move
            inputs = obs[:84].reshape(8064)
            action_idx = np.argmax(net.activate(inputs))
            if collision(obs, i):
                print("Fitness: {:.2f}".format(g.fitness))
                print("Score: {:.2f}".format(score))
                break
            

        obs_, reward, _, _ = env.step(actions[action_idx])
        env.render()
        obs = rgb2gray(obs_)

        score += reward
        g.fitness += actions[action_idx][1] - actions[action_idx][2]
        g.fitness -= 1

        if collision(obs, i):
            print("Fitness: {:.2f}".format(g.fitness))
            print("Score: {:.2f}".format(score))
            break

        scores.append(score)


def run(config_file):
    print('Generating genetic matter...')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 100)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    print('Starting up!')
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, "config.txt")
    # run(config_path)

    test()
