#!/usr/bin/env python3
import gym
import numpy as np
import random
from meta_nn.meta_nn import MetaNet


env = gym.make("CliffWalking-v0")
lr = 0.1
df = 1.0
eps = 0.5
Q = np.zeros((env.nS, env.nA))

replay_buffer = []
rewards = np.zeros((1000,))

decay_factor = (0.01 / eps) ** (1 / 1000)

for i in range(1000):
    
    done = False
    state = env.reset()

    while not done:
        if random.uniform(0,1) < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + lr * (reward + np.max(Q[next_state]) - Q[state, action])

        replay_buffer.append((state, action, next_state))
        state = next_state
        rewards[i] += reward

    eps = eps * decay_factor

print(rewards)
print(replay_buffer)
