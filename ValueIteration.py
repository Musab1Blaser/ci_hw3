import numpy as np
import gymnasium as gym
import random
import math

env = gym.make("Taxi-v3", render_mode="human")  

# change-able parameters:
discount_factor = 0.9
delta_threshold = 0.00000001
# epsilon = 1 # ???

def value_iteration(env, discount_factor, delta_threshold):

    def find_max(state_rewards, V):

        value = -math.inf
        index = 0

        for i in range(len(state_rewards)):

            d = discount_factor
            if V[state_rewards[i][0]] < 0: # if the value of the state is negative, it becomes MORE negatve after discounting
                d = 2 - d

            if value < (state_rewards[i][1] + d * V[state_rewards[i][0]]):
                value = state_rewards[i][1] + d * V[state_rewards[i][0]]
                index = i

        return value, index

    num_states = env.observation_space.n
    # num_actions = env.action_space.n

    # Initialize the value function
    V = np.zeros(num_states)

    # For each state, the policy will tell you the action to take
    policy = np.zeros(num_states, dtype=int)

    #Write your code to implement value iteration main loop

    env_unwrapped = env.unwrapped # saw this in some random discussion post online, it's necessary to use the P attribute

    while True:

        delta = 0

        for state in range(num_states):
            actions = env_unwrapped.P[state]
            state_rewards = [(actions[i][0][1], actions[i][0][2]) for i in range(len(actions))]
            # print(state_rewards)
            v = V[state]
            V[state], policy[state] = find_max(state_rewards, V) # find max value and action index
            delta = max(delta, abs(v - V[state]))

        # print("Delta:", delta)
        if delta < delta_threshold:
            break

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
    # action = env.action_space.sample() # random action for testing
    new_state, reward, done, truncated, info = env.step(action) # information after taking the action
    env.render()
    if done:
        print("number of steps taken:", step)
        break

    state = new_state
    
env.close()