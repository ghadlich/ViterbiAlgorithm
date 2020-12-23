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
from generate_observations import generate_observations

def print_result(observed, actual, viterbi, max_length=60):
    ''' Pretty Prints the Output '''

    # Chop it into chunks
    obs_chunks = [observed[i:i+max_length] for i in range (0, len(observed), max_length)]
    act_chunks = [actual[i:i+max_length] for i in range (0, len(actual), max_length)]
    vit_chunks = [viterbi[i:i+max_length] for i in range (0, len(viterbi), max_length)]

    # Print out the chunks of max_length
    for i in range(len(obs_chunks)):
        print(f'{"Observed":9}: {obs_chunks[i]}')
        print(f'{"Actual":9}: {act_chunks[i]}')
        print(f'{"Viterbi":9}: {vit_chunks[i]}')
        print("")

if __name__ == "__main__":
    # Loaded Die Flip Example
    # There are n dice rolls, with a die flipping between loaded and fair
    # State 0 = Fair
    # State 1 = Loaded

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

    # Execute Viterbi Algorithm
    v = ViterbiAlgorithm.ViterbiAlgorithm(transition_matrix, emission_matrix, initial_state_probability, 
                                          dice_rolls, 
                                          state_space=['F', 'L'],
                                          observation_space=['1', '2', '3', '4', '5', '6'],
                                          use_log_probabilities=True)
    v.run()

    # Convert Output To String
    # Convert to Strings
    true_state = ''.join(str(i) for i in true_state)
    dice_rolls = ''.join(str(i) for i in dice_rolls)
    viterbi    = ''.join(str(i) for i in v.viterbi_path)

    # Print out result
    print_result(dice_rolls, true_state, viterbi)
