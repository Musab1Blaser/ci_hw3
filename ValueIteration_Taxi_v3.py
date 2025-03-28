import numpy as np
import gym
import random
import math


env = gym.make("Taxi-v3", render_mode="human")  

# change-able parameters:
discount_factor = 0.8
delta_threshold = 0.00001
epsilon = 1


def value_iteration(env, gamma, epsilon):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the value function
    V = np.zeros(num_states)

    #Write your code to implement value iteration main loop
    


    # For each state, the policy will tell you the action to take
    policy = np.zeros(num_states, dtype=int)

    # Write your code here to extract the optimal policy from value function.



    return policy, V


# Run value iteration
policy, V = value_iteration(env, discount_factor, delta_threshold)


# resetting the environment and executing the policy
state = env.reset()
#state = state[0]
step = 0
done = False
state = state[0]

max_steps = 100
for step in range(max_steps):

    # Getting max value against that state, so that we choose that action
  
    action = policy[state]
    new_state, reward, done, truncated, info = env.step(action) # information after taking the action
    env.render()
    if done:
        print("number of steps taken:", step)
        break

    state = new_state
    
env.close()