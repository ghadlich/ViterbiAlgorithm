# Example Viterbi Implementation

This package includes a python / numpy implementation to find the Viterbi Path of an input set of observations. This is useful when dealing with Hidden Markov Models.

### Generating Synthetic Observations
The generate_observations .py file takes in an initial probability, transition matrix, and emissions matrix to generate synthetic observations.

```
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
```

Output:
```
$ python generate_observations.py

Rolls   : 453521123164266656412446615666665444621516153163662666666436
Actual  : FFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLFFFFFFFFFFFFFFLLLLLLLLLLLLLL

Rolls   : 215466646561664242124651212224225422545115322161655131312216
Actual  : LLFLLLLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

Rolls   : 353245565163614165255512335126565264624343664131653133463636
Actual  : FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLL

Rolls   : 616464666651231154666531254315654541152461665513656641614366
Actual  : LLLLLLLLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFFLLLLLLFFFLLLLFFFFFFFFF

Rolls   : 642545611166666366446616266662554646212346453256561435435121
Actual  : FFFFFFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFF
```

### Example Viterbi Path
In the example_viterby.py file, a 6-sided die is rolled while being interchanged with a loaded die. The loaded die has an affinity for 6.

__Initial State Probability:__

| Transitions   |   Fair    |   Loaded  |
|---------------|-----------|-----------|
| Begin         |    0.9999 |    0.0001 |

__Transition Matrix:__

| Transitions   |   Fair    |   Loaded  |
|---------------|-----------|-----------|
| Fair          |    0.95   |    0.05   |
| Loaded        |    0.10   |    0.90   |

__Emission Matrix:__

| Emissions   |    1 |    2 |    3 |    4 |    5 |    6 |
|-------------|------|------|------|------|------|------|
| Fair        | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| Loaded      | 0.2  | 0.3  | 0.3  | 0.2  | 0.25 | 0.25 |

```
$ python example_viterbi.py

Observed : 453521123164266656412446615666665444621516153163662666666436
Actual   : FFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLFFFFFFFFFFFFFFLLLLLLLLLLLLLL
Viterbi  : FFFFFFFFFFFFFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL

Observed : 215466646561664242124651212224225422545115322161655131312216
Actual   : LLFLLLLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
Viterbi  : LLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

Observed : 353245565163614165255512335126565264624343664131653133463636
Actual   : FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLL
Viterbi  : FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLL

Observed : 616464666651231154666531254315654541152461665513656641614366
Actual   : LLLLLLLLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFFLLLLLLFFFLLLLFFFFFFFFF
Viterbi  : LLLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLLLLLLL

Observed : 642545611166666366446616266662554646212346453256561435435121
Actual   : FFFFFFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFF
Viterbi  : LLLLLLLLLLLLLLLLLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
```

### Estimating Transition and Emission Matrices from Observations
In estimate_based_on_observation.py, a loaded die is rolled 1,000,000 times and the viterbi algorithm is used to estimate the transition and emission matrices.

```
$ python estimate_based_on_observation.py

```

__Truth:__

| Transitions   |   Fair |   Loaded |
|---------------|--------|----------|
| Fair          |   0.95 |     0.05 |
| Loaded        |   0.05 |     0.95 |

| Emissions   |        1 |        2 |        3 |        4 |        5 |        6 |
|-------------|----------|----------|----------|----------|----------|----------|
| Fair        | 0.166667 | 0.166667 | 0.166667 | 0.166667 | 0.166667 | 0.166667 |
| Loaded      | 0.04     | 0.04     | 0.04     | 0.04     | 0.04     | 0.8      |


__Estimated After 11 Runs:__

| Transitions   |      Fair |   Loaded |
|---------------|-----------|----------|
| Fair          | 0.970688  | 0.029312 |
| Loaded        | 0.0283057 | 0.971694 |

| Emissions   |         1 |        2 |        3 |         4 |         5 |        6 |
|-------------|-----------|----------|----------|-----------|-----------|----------|
| Fair        | 0.167174  | 0.166878 | 0.167176 | 0.168417  | 0.168816  | 0.161539 |
| Loaded      | 0.0419376 | 0.041629 | 0.04249  | 0.0412889 | 0.0415386 | 0.791116 |