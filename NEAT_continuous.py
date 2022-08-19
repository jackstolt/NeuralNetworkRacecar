from re import L
from processing import rgb2gray, collision, lidar
import itertools as it
import numpy as np
import pickle
import neat
import gym
import sys
import os

CHECKPOINT = False
RENDER = False
VERBOSE = False
TEST = False

def print_obs(obs, x_min=0, x_max=96, y_min=0, y_max=96):
    '''
    Print a specific section of an observation
    '''
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            print('{:.0f} '.format(obs[y][x]), end='')
        print()
def test():
    '''
    Put in place to test mechanics of the game
    '''
    print('Start Game!')
    env = gym.make('CarRacing-v1', )
    ep = 0
    obs = env.reset()

    action = [0,1,0]

    obs = rgb2gray(obs)
    print_obs(obs, 0,96,0,84)
    print_obs(obs, 45,51,66,77)

    while not collision(obs, ep):
        ep += 1
        print(collision(obs, ep))
        obs, _, _, _ = env.step(action)
        obs = rgb2gray(obs)
        env.render()

        print(ep)


def eval_genomes(genomes, config):
    '''
    Evaluates the genomes based on the config file settings.
    genomes -> are the set genomes for a given population to be tests
    config -> the processed config file
    '''
    print('Evaluating genomes...')
    env = gym.make('CarRacing-v1', verbose=False)

    scores = []
    for id, g in genomes:
        score = 0
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)

        obs = env.reset()
        obs = rgb2gray(obs)
        level = 0
        for step in range(1000):
            # determine action based on observation
            output = net.activate(lidar(obs, env))
            speed = lidar(obs, env)[0]
            
            # process action
            turn_output = output[0]
            gas_output = 1 if speed < 50 else .1

            action = (turn_output, gas_output, 0)

            # make action
            obs_, reward, _, _ = env.step(action)
            obs = rgb2gray(obs_)
            if RENDER:
                env.render()
            if VERBOSE:
                print(action)
                print('step {} -> speed {:.2f}'.format(step, speed))

            g.fitness += reward

            score += reward
            if score // 50 != level:             # ever increasing checkpoint rewards
                g.fitness += 50
                level = score // 50

            if step > 50 and speed < 5:
                g.fitness -= 100
                print('Genome -> {}\tFitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))
                break
            if collision(obs, step):
                g.fitness -= 100
                print('Genome -> {} Fitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))
                break
            if step == 999:
                print('TIMED OUT!')
                print('Genome -> {} Fitness -> {:.2f}\tScore -> {:.2f}'.format(id, g.fitness, score))

        scores.append(score)

def run(config):
    '''
    Actually runs a generation and calls the eval_genomes function to evaluate it
    config -> the desired config file
    '''
    print('Generating genetic matter...')   

    if CHECKPOINT:
        p = neat.Checkpointer.restore_checkpoint('continuous-checkpoint-')
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    best = p.run(eval_genomes, 100)

    print('\nBest genome:\n{!s}'.format(best))


def test_agent(config, file='best.pickle'):
    with open(file, 'rb') as f:
        best = pickle.load(f)

    env = gym.make('CarRacing-v1') 
    step = 0
    score = 0

    net = neat.nn.FeedForwardNetwork.create(best, config)
    obs = env.reset()
    obs = rgb2gray(obs)

    for t in range(1000):
        # determine action based on observation
        output = net.activate(lidar(obs, env))
        speed = lidar(obs, env)[0]

        # process action
        turn_output = output[0]
        gas_output = 1 if speed < 50 else .1

        action = (turn_output, gas_output, 0)

        # make action
        obs_, reward, _, _ = env.step(action)
        obs = rgb2gray(obs_)
        env.render()
        
        score += reward
        if collision(obs, step):
            print('Score -> {:.2f}'.format(score))
            break
        step += 1
    print('Score -> {:.2f}'.format(score))
    return score


if __name__ == '__main__':
    print('Starting up!')

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_continuous.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    for arg in sys.argv:
        if(arg == '-r'):
            RENDER = True
        if(arg == '-c'):
            CHECKPOINT = True
        if(arg == '-v'):
            VERBOSE = True
        if(arg == '-t'):
            scores = []
            for _ in range(5):
                scores.append(test_agent(config))
            print('Average Score -> {:.2f}'.format(np.mean(scores)))
            exit(0)
        if(arg == '-d'):
            test()
            exit(0)

    run(config)