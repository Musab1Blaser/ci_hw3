import numpy as np
import gymnasium as gym
import math

env = gym.make("Taxi-v3", render_mode="human")  

def value_iteration(env, discount_factor, delta_threshold):

    def find_max(state_rewards, V): # find the max value action from a given state

        value = -math.inf # initialize to negative infinity
        index = 0

        for i in range(len(state_rewards)):

            d = discount_factor 
            # if V[state_rewards[i][0]] < 0: # If the value of the state is negative, it becomes MORE negatve after discounting. Results are similar with or without this.
            #     d = 2 - d

            if value < (state_rewards[i][1] + d * V[state_rewards[i][0]]):
                value = state_rewards[i][1] + d * V[state_rewards[i][0]]
                index = i

        return index, value

    num_states = env.observation_space.n

    # Initialize the value function
    V = np.zeros(num_states)

    # For each state, the policy will tell you the action to take
    policy = np.zeros(num_states, dtype=int)

    env_unwrapped = env.unwrapped # saw this in some random discussion post online, it's necessary to use the P attribute at line 48

    while True:

        delta = 0 # Initialize delta to 0 for each iteration

        for state in range(num_states): # for each state 

            actions = env_unwrapped.P[state] # get all actions
            state_rewards = [(actions[i][0][1], actions[i][0][2]) for i in range(len(actions))] # get the states and rewards for each action from a certain state and store it as a (state, reward) tuple
            
            v = V[state] # store the current value of the state
            policy[state], V[state] = find_max(state_rewards, V) # update the value and policy
            delta = max(delta, abs(v - V[state])) # update delta to the max difference between the old and new value of the state

        if delta < delta_threshold: # if there were no changes bigger than the threshold, we can stop the iteration
            break

    return policy, V

discount_factor = 0.9
delta_threshold = 0.00000001

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