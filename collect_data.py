import noise
import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

env = gym.make("CarRacing-v0")
from pyglet import gl
import cProfile
_ = env.reset()

from noise import pnoise1
class Run:
    def __init__(self):
        self.obs_l = []
        self.reward_l = []
        self.action_l = []
    
def makeActionArray(time_scale = 200, magnitude_scale=1.7):
    start = np.random.random(4) * 10000
    out = []
    for _ in range(2000):
        action = [ magnitude_scale * pnoise1(_ / time_scale + start_i, 5) for start_i in start]
        out.append(action)
    return np.array(out)


def randomRun(sparse=False, encoded=False, visualize=False):
    _ = env.reset()
    done = False
    run = Run()

    run.action_l = makeActionArray()
    #action = env.action_space.sample()
    i = 0
    while not done:
        action = run.action_l[i]
        i += 1
        obs, reward, done, _ = env.step(action)
        #action_l.append(action)
        if(i % 10 == 0 or not sparse):
            run.obs_l.append(obs)
        run.reward_l.append(reward)
        
        if (len(run.reward_l) % 20 == 0) and visualize:
            env.render()
    if encoded:
        run.obs_l = vae.encoder.predict(np.array(run.obs_l) / 255.)
    return run
    
    
if __name__ == "__main__":
    
    runs = []
    for _ in range(10):
        print("main2")
        runs.append(randomRun(encoded=False, sparse=False))
    with open("output_file.pickle", "wb") as f:
        pickle.dump(runs, f)
    
