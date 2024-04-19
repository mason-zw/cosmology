import numpy as np


def proposal(x, step_size=1.0):
    """ Generate a proposal given the current state x. """
    return x + step_size * np.random.randn()

def run_mcmc(n_samples, initial_state, log_target_distribution, step_size=1.0):
    """ Metropolis-Hastings MCMC algorithm to sample from the target distribution using log probabilities. """
    samples = np.zeros(n_samples)
    current_state = initial_state
    current_log_prob = log_target_distribution(current_state)
    
    for i in range(n_samples):
        proposed_state = proposal(current_state, step_size)
        proposed_log_prob = log_target_distribution(proposed_state)
        
        # Calculate acceptance probability in log space
        acceptance_log_prob = proposed_log_prob - current_log_prob
        
        # Accept or reject the proposed state
        if np.log(np.random.rand()) < acceptance_log_prob:
            current_state = proposed_state
            current_log_prob = proposed_log_prob
        
        samples[i] = current_state
    
    return samples


######### affine version #########

def affine_invariant_proposal(current, complement, a=2.0):
    """ Generate a proposal using affine invariant ensemble method. """
    r = np.random.rand() * (a - 1/a) + 1/a
    return complement + r * (current - complement)


def run_affine_mcmc(n_samples, n_walkers, initial_states, log_target_distribution, a=2.0):
    """ Ensemble Metropolis-Hastings MCMC using affine invariant proposals. """
    samples = np.zeros((n_samples, n_walkers, len(initial_states[0])))
    current_states = np.array(initial_states)
    current_log_probs = np.array([log_target_distribution(x) for x in current_states])
    
    for i in range(n_samples):
        for j in range(n_walkers):
            # Select a random walker that is not the current walker
            others = list(range(n_walkers))
            others.remove(j)
            k = np.random.choice(others)
            
            proposed_state = affine_invariant_proposal(current_states[j], current_states[k], a)
            proposed_log_prob = log_target_distribution(proposed_state)
            
            # Calculate acceptance probability in log space
            acceptance_log_prob = proposed_log_prob - current_log_probs[j]
            
            # Accept or reject the proposed state
            if np.log(np.random.rand()) < acceptance_log_prob:
                current_states[j] = proposed_state
                current_log_probs[j] = proposed_log_prob
            
            samples[i, j] = current_states[j]
    
    return samples