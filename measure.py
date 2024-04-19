import numpy as np
from mcmc import run_affine_mcmc
from github.cosmology_emcee import *
import corner  
import matplotlib.pyplot as plt

data = ascii.read('apj341188t13_mrt.txt', format='mrt')
z, mB, mB_err, x1, x1_err, cc, c_err = data['zCMB'], data['Bmag'], data['e_Bmag'], data['x1'], data['e_x1'], data['c'], data['e_c']


def lnprob(p):
    H0, OmegaM, OmegaDE, w, alpha, beta, M = p
    mu = muB(mB, x1, cc, alpha, beta, M)
    mu_err = muB_err(mB_err, x1_err, c_err, alpha, beta)
    if np.isnan(-0.5 * np.sum((mu - dist_modulus(z, H0, OmegaM, OmegaDE, w))**2 / mu_err**2)):
        return print(p)
    return -0.5 * np.sum((mu - dist_modulus(z, H0, OmegaM, OmegaDE, w))**2 / mu_err**2)


params = ['H0', 'OmegaM', 'OmegaDE', 'w', 'alpha', 'beta', 'M']
nwalkers = 32
ndim = len(params)
p0 = np.random.normal(1, 0.1, (nwalkers, ndim)) * np.array([70, 0.3, 0.7, -1, 0.112, 3.1, -19])

n_steps = 3000
samples = run_affine_mcmc(n_steps, nwalkers, p0, lnprob)

corner.corner(samples.reshape(n_steps*nwalkers, len(p0)), labels=params)
plt.savefig('corner.png', dpi=300)
np.save('samples.npy', samples)