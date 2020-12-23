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

class ViterbiAlgorithm(object):
    """A class that implements the Viterbi Algorithm"""

    def __init__(self, transition_matrix_a, emission_matrix_b, initial_state_array_i, observed_array_o, state_space=None, observation_space=None, use_log_probabilities=False):
        """ Intialize Local Variables """

        # Save Inputs
        self._state_space = state_space

        if observation_space == None:
            self._observation_space = [i for i in range(len(emission_matrix_b[0]))]
        else:
            self._observation_space = observation_space
        
        # Setup Index Lookup for Observations
        self._obs_to_index = dict()
        for i in range(len(self._observation_space)):
            self._obs_to_index[self._observation_space[i]] = i

        self._use_log_probabilities = use_log_probabilities
        self._transition_matrix_a = np.array(transition_matrix_a)
        self._emission_matrix_b = np.array(emission_matrix_b)
        self._initial_state_array_i = np.array(initial_state_array_i)

        # Replace Observations with indices
        if observation_space != None:
            self._observed_array_o = np.array([self._obs_to_index[obs] for obs in observed_array_o])
        else:
            self._observed_array_o = np.array(observed_array_o)

        # If using log probabilities, initialize matrices with small deltas to protect against zero
        if (self._use_log_probabilities):
            self._transition_matrix_a = np.log(self._transition_matrix_a + np.finfo(float).eps)
            self._emission_matrix_b = np.log(self._emission_matrix_b + np.finfo(float).eps)
            self._initial_state_array_i = np.log(self._initial_state_array_i + np.finfo(float).eps)

        # Retrieve Sizes
        self._num_states = self._transition_matrix_a.shape[0]
        self._num_observations = len(self._observed_array_o)

        # Initialize Helpers
        self._accumlated_probability_matrix = np.zeros((self._num_states, self._num_observations))
        self._backtrack_matrix = np.zeros((self._num_states, self._num_observations-1), dtype=int)

        # Initialize Accumulated Probability Matrix
        self._accumlated_probability_matrix[:, 0] = self._initial_state_array_i + \
                                                 self._emission_matrix_b[:, 0]

        # Create Optimial State Sequence
        self.viterbi_path = []
        self.viterbi_probability = 0
        return


    def run(self):
        '''Execute Viterbi Algorithm'''

        # Compute Accumulated Probability and Backtrack Matrices
        for i in range(1,self._num_observations):
            for j in range(self._num_states):
                # Use Logs if requested
                if (self._use_log_probabilities):
                    product = self._transition_matrix_a[:, j] + \
                              self._accumlated_probability_matrix[:, i-1]

                    self._accumlated_probability_matrix[j, i] = \
                        np.max(product) + self._emission_matrix_b[j, self._observed_array_o[i]]
                else:
                    product = np.multiply(self._transition_matrix_a[:, j], \
                                          self._accumlated_probability_matrix[:, i-1])

                    self._accumlated_probability_matrix[j, i] = \
                        np.max(product) * self._emission_matrix_b[j, self._observed_array_o[i]]

                # Update Backtract Matrix
                self._backtrack_matrix[j, i-1] = np.argmax(product)

        # Set Initial Path to 0
        self.viterbi_path = np.zeros(self._num_observations, dtype=int)

        # Set last entry to most likely state
        self.viterbi_path[-1] = np.argmax(self._accumlated_probability_matrix[:, -1])

        # Set Viterbi Probability
        self.viterbi_probability = self._accumlated_probability_matrix[:, -1][self.viterbi_path[-1]]

        # Starting from the end-1 to backtrack viterbi path
        for i in range(self._num_observations-2, 0, -1):
            self.viterbi_path[i] = \
                self._backtrack_matrix[int(self.viterbi_path[i+1]), i]

        # If state space was provided, convert viterbi path to state space
        if self._state_space != None:
            self.viterbi_path = [self._state_space[self.viterbi_path[i]] for i in range(len(self.viterbi_path))]

        return

if __name__ == "__main__":
    print(__file__)
