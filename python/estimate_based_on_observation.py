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
import os
import numpy as np
import ViterbiAlgorithm
from tabulate import tabulate
from generate_observations import generate_observations

def pretty_print(initial_state_probability, transition_matrix, emission_matrix, state_space, observation_space):
    ''' Pretty Prints the initialized matrices '''

    header = ["Transitions"]
    data = [["Begin"]]
    for i in range(len(initial_state_probability)):
        header.append(state_space[i])
        data[0].append(str(initial_state_probability[i]))

    print(tabulate(data, header, tablefmt="github"))
    print("")

    # Transition Matrix
    # Header is same as before
    data = []

    for i in range(len(transition_matrix)):
        row = []
        row.append(state_space[i])
        for j in range(len(transition_matrix[i])):
            row.append(str(transition_matrix[i][j]))

        data.append(row)

    print(tabulate(data, header, tablefmt="github"))
    print("")

    # Emissions Matrix
    header = ["Emissions"]
    for i in range(len(emission_matrix[0])):
        header.append(observation_space[i])
    data = []

    for i in range(len(emission_matrix)):
        row = []
        row.append(state_space[i])
        for j in range(len(emission_matrix[i])):
            row.append(str(emission_matrix[i][j]))

        data.append(row)

    print(tabulate(data, header, tablefmt="github"))
    print("")

    return

def updateMatricesAfterRun(calculated_states, observations, state_space, observation_space):
    """ Updates Transition and Emission Matrices based on observations """

    # Create New Transition Matrix
    num_states = len(state_space)

    # Setup Index Lookup for States
    state_to_index = dict()
    for i in range(len(state_space)):
        state_to_index[state_space[i]] = i

    # Init Transition Matrix
    transitionMatrix = np.zeros((num_states,num_states))
    for i in range(1, len(calculated_states)):
        transitionMatrix[state_to_index[calculated_states[i-1]]][state_to_index[calculated_states[i]]] += 1

    for i in range(len(transitionMatrix)):
        transitionMatrix[i] /= np.sum(transitionMatrix[i])

    # Create New Emissions Matrix
    num_observations = len(observation_space)

    # Setup Index Lookup for Observations
    obs_to_index = dict()
    for i in range(len(observation_space)):
        obs_to_index[observation_space[i]] = i

    emissionMatrix = np.zeros((num_states, num_observations))
    for i in range(len(observations)):
        emissionMatrix[state_to_index[calculated_states[i]]][obs_to_index[observations[i]]] += 1

    for i in range(len(emissionMatrix)):
        emissionMatrix[i] /= np.sum(emissionMatrix[i])

    return transitionMatrix, emissionMatrix

def runLoop(initial_state_probability, transition_matrix, emission_matrix, observations, state_space, observation_space, runs=100):
    # Execute Viterbi Algorithm and recalculate transition and emission matrices

    previous_probability = 0
    for i in range(runs):
        v = ViterbiAlgorithm.ViterbiAlgorithm(transition_matrix, emission_matrix, initial_state_probability, 
                                              observations, 
                                              state_space=state_space, 
                                              observation_space=observation_space,
                                              use_log_probabilities=True)
        v.run()

        print("Run " + str(i+1) + " Probability: " + str(v.viterbi_probability))
        
        if (abs(previous_probability-v.viterbi_probability) < 1e-6):
            print("Converged after " + str(i+1) + " Runs")
            break

        previous_probability = v.viterbi_probability

        transition_matrix, emission_matrix = updateMatricesAfterRun(v.viterbi_path, observations, state_space, observation_space)

    return transition_matrix, emission_matrix

if __name__ == "__main__":
    print(__file__)

    # Initial State Probability
    initial_state_probability = [0.9999, 0.0001]

    # Transition Matrix
    transition_matrix = [[0.95, 0.05],\
                        [0.05, 0.95]]

    # Emission Matrix
    emission_matrix = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],\
                      [0.04, 0.04, 0.04, 0.04, 0.04, 0.8]]

    state_space=['Fair','Loaded']
    observation_space=['1', '2', '3', '4', '5', '6']

    print("Truth:")
    pretty_print(initial_state_probability, transition_matrix, emission_matrix, state_space, observation_space)

    true_state, dice_rolls = generate_observations(initial_state_probability, transition_matrix, emission_matrix,
                                            num_samples=1000000,
                                            state_space=state_space,
                                            observation_space=observation_space,
                                            seed=515)

    # Transition Matrix
    estimated_transition_matrix = [[0.95, 0.05],\
                                   [0.10, 0.90]]

    # Emission Matrix
    estimated_emission_matrix = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],\
                                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]

    estimated_transition_matrix, estimated_emission_matrix = \
        runLoop(initial_state_probability, estimated_transition_matrix, estimated_emission_matrix, 
                dice_rolls,
                state_space=state_space,
                observation_space=observation_space,
                runs=15)

    print("\nEstimated: ")
    pretty_print(initial_state_probability, estimated_transition_matrix, estimated_emission_matrix, state_space, observation_space)
