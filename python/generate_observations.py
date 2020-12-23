#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2020 Grant Hadlich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE. 
import numpy as np


def _check_input(initial_state_probability, transition_matrix, emission_matrix, num_states, num_observations):
    """ Check Inputs for Consistency """
    if len(initial_state_probability) != num_states:
        print("Error: Number of States does not match provided Initial State Probabilities")
        return False

    if len(transition_matrix) != num_states:
        print("Error: Number of States does not match provided Transition Matrix")
        return False

    for i in range(len(transition_matrix)):
        if len(transition_matrix[i]) != num_states:
            print("Error: Number of States does not match provided Transition Matrix Row: " + str(i))
            return False

    if len(emission_matrix) != num_states:
        print("Error: Number of States does not match provided Emission Matrix")
        return False

    for i in range(len(emission_matrix)):
        if len(emission_matrix[i]) != num_observations:
            print("Error: Observation Space does not match provided Emission Matrix Row: " + str(i))
            return False

    return True

def generate_observations(initial_state_probability, transition_matrix, emission_matrix, num_samples=300, state_space=None, observation_space=None, seed=None):
    """Generates an observation and associated ground truth

    Loaded Die Example:
        initial_state_probability = [0.9999, 0.0001]

        transition_matrix = [[0.95, 0.05],
                            [0.10, 0.90]]

        emission_matrix = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]

        state_space =['F','L']

        observation_space=['1', '2', '3', '4', '5', '6']

    Args:
        initial_state_probability (1xn float list): Probability that state n is the first state
        transition_matrix (nxn float list): Probability that state n+1 is selected after state n
        emission_matrix (nxo float list): Probability that observation o is observed in state n
        num_samples (int): Number of observations to observe
        state_space (list): States representation
        observation_space (list): Observation representation
        seed (int): seed used in random function

    Returns:
        states (list): The true states during observation
        observations (list): The observation at each state

    """
    if state_space == None:
        state_space = [i for i in range(len(initial_state_probability))]

    if observation_space == None:
        observation_space = [i for i in range(len(emission_matrix[0]))]

    num_states = len(state_space)
    num_observations = len(observation_space)

    if not _check_input(initial_state_probability, transition_matrix, emission_matrix, num_states, num_observations):
        return None, None

    # Set Seed if one exists
    np.random.seed(seed)

    # Setup Index Lookup for States
    state_to_index = dict()
    for i in range(len(state_space)):
        state_to_index[state_space[i]] = i
    
    # Setup Index Lookup for Observations
    obs_to_index = dict()
    for i in range(len(observation_space)):
        obs_to_index[observation_space[i]] = i

    # Find Initial State
    state = np.random.choice(state_space, 1, p=initial_state_probability)[0]
    observation = np.random.choice(observation_space, 1, p=emission_matrix[state_to_index[state]])[0]

    # Initialize Returns
    states_return = [state]
    obs_return = [observation]

    # Generate the Observations
    for _ in range(num_samples-1):
        state = np.random.choice(state_space, 1, p=transition_matrix[state_to_index[state]])[0]
        observation = np.random.choice(observation_space, 1, p=emission_matrix[state_to_index[state]])[0]

        states_return.append(state)
        obs_return.append(observation)

    return states_return, obs_return

def print_result(observed, actual, max_length=60):
    ''' Pretty Prints the Output '''

    # Chop it into chunks
    obs_chunks = [observed[i:i+max_length] for i in range (0, len(observed), max_length)]
    act_chunks = [actual[i:i+max_length] for i in range (0, len(actual), max_length)]

    # Print out the chunks of max_length
    for i in range(len(obs_chunks)):
        print(f'{"Rolls":8}: {obs_chunks[i]}')
        print(f'{"Actual":8}: {act_chunks[i]}')
        print("")

if __name__ == "__main__":
    # Loaded Die Flip Example
    # There are n dice rolls, with a die flipping between loaded and fair

    # Initial State Probability
    initial_state_probability = [0.9999, 0.0001]

    # Transition Matrix
    transition_matrix = [[0.95, 0.05],\
                        [0.10, 0.90]]

    # Emission Matrix
    emission_matrix = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],\
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]

    true_state, dice_rolls = generate_observations(initial_state_probability, transition_matrix, emission_matrix,
                                            num_samples=300,
                                            state_space=['F','L'],
                                            observation_space=['1', '2', '3', '4', '5', '6'],
                                            seed=515)

    # Convert to Strings
    true_state = ''.join(str(i) for i in true_state)
    dice_rolls = ''.join(str(i) for i in dice_rolls)

    # Print out result
    print_result(dice_rolls, true_state)
