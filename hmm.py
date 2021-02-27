from list_utils import vector_scalar_mult



class HMM:
    '''
    Hidden Markov Model with parameters.
    Order of axes: time, from_state, state, to_state, emission
    '''
    def __init__(self, initial_distribution, transition_probabilities, emission_probabilities):
        self.num_states = len(initial_distribution)
        self.num_emissions = len(emission_probabilities[0])
        self.initial_distribution = initial_distribution
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
    

    def improve(self, emission_sequence):
        '''Returns an improved model given an emission sequence'''
        gamma, di_gamma = self.gammas(emission_sequence=emission_sequence)
        initial_distribution = gamma[0]

        transition_probabilities = []
        for from_state in range(self.num_states):
            transition_probabilities.append([])
            denominator = sum(gamma_moment[from_state] for gamma_moment in gamma[:-1])
            for to_state in range(self.num_states):
                numerator = sum(di_moment[from_state][to_state] for di_moment in di_gamma)
                transition_probabilities[from_state].append(numerator / denominator)
        
        emission_probabilities = []
        for state in range(self.num_states):
            emission_probabilities.append([])
            denominator = sum(gamma_moment[state] for gamma_moment in gamma[:-1])
            for emission in range(self.num_emissions):
                numerator = sum(
                    gamma_moment[state]
                    for time, gamma_moment in enumerate(gamma)
                    if emission_sequence[time] == emission
                )
                emission_probabilities[state].append(numerator / denominator)
        
        return type(self)(
            initial_distribution=initial_distribution,
            transition_probabilities=transition_probabilities,
            emission_probabilities=emission_probabilities
        )
    

    def transition_step(self, state_distribution):
        '''Returns the distribution over states after one step, starting from state_distribution'''
        return [
            sum(
                state_distribution[from_state] * self.transition_probabilities[from_state][to_state]
                for from_state in range(self.num_states)
            )
            for to_state in range(self.num_states)
        ]
    

    def emissions_probability(self, emission_sequence):
        '''Returns the probability that this model produced the given sequence of emissions'''
        return sum(self.forward_pass(emission_sequence)[-1])
    

    def most_probable_states(self, emission_sequence):
        '''Returns the most likely state sequence given the emissions'''
        deltas = self._alpha_delta_init(emission_sequence)
        prev_states = []
        for emission in emission_sequence[1:]:
            choices = [
                [
                    prev_delta * self.transition_probabilities[prev_state][state]
                    for prev_state, prev_delta in enumerate(deltas[-1])
                ]
                for state in range(self.num_states)
            ]
            chosen = [max(choice) for choice in choices]
            prev_states.append([choice.index(chosen_value) for choice, chosen_value in zip(choices, chosen)])
            deltas.append([c * emit[emission] for c, emit in zip(chosen, self.emission_probabilities)])
        best_path_probability = max(deltas[-1])
        best_path = [deltas[-1].index(best_path_probability)]
        for possible_prev_states in reversed(prev_states):
            best_path = [possible_prev_states[best_path[0]]] + best_path # FIXME: prepending is slow af
        return best_path
    

    def forward_pass(self, emission_sequence, scale=False):
        '''Returns the sequence of alphas'''
        alphas = self._alpha_delta_init(emission_sequence)
        if scale:
            scaling_factors = [1 / sum(alphas[0])]
            alphas[0] = vector_scalar_mult(alphas[0], scaling_factors[0])
        for emission in emission_sequence[1:]:
            priors = self.transition_step(alphas[-1])
            alphas.append([prior * emit[emission] for prior, emit in zip(priors, self.emission_probabilities)])
            if scale:
                scaling_factors.append(1 / sum(alphas[-1]))
                alphas[-1] = vector_scalar_mult(alphas[-1], scaling_factors[-1])
        if scale:
            return alphas, scaling_factors
        return alphas
    

    def backward_pass(self, emission_sequence, scaling_factors=None):
        '''Returns the sequence of betas'''
        betas = [None] * len(emission_sequence)
        betas[-1] = [scaling_factors[-1] if scaling_factors is not None else 1] * self.num_states

        # For time iteration, go backwards in the emission sequence, skipping time = T
        for time in range(len(emission_sequence) - 2, -1, -1):
            betas[time] = [
                sum(
                    self.transition_probabilities[from_state][to_state] *
                    self.emission_probabilities[to_state][emission_sequence[time + 1]] *
                    betas[time + 1][to_state] *
                    (scaling_factors[time] if scaling_factors is not None else 1)
                    for to_state in range(self.num_states)
                )
                for from_state in range(self.num_states)
            ]
        return betas


    def gammas(self, emission_sequence):
        '''Returns the gamma and di-gamma sequences needed for parameter estimation'''
        alphas, scaling_factors = self.forward_pass(emission_sequence, scale=True)
        betas = self.backward_pass(emission_sequence, scaling_factors=scaling_factors)
        gamma = []
        di_gamma = []

        for time in range(len(emission_sequence) - 1):
            gamma.append([])
            di_gamma.append([])
            for from_state in range(self.num_states):
                di_gamma[time].append([])
                for to_state in range(self.num_states):
                    di_gamma[time][from_state].append(
                        alphas[time][from_state] *
                        self.transition_probabilities[from_state][to_state] *
                        self.emission_probabilities[to_state][emission_sequence[time + 1]] *
                        betas[time + 1][to_state]
                    )
                gamma[time].append(sum(di_gamma[time][from_state]))

        gamma.append(alphas[-1])
        return gamma, di_gamma


    def _alpha_delta_init(self, emission_sequence):
        return [[
            init * emit[emission_sequence[0]]
            for init, emit in zip(self.initial_distribution, self.emission_probabilities)
        ]]