# The version using emcee instead of my own MCMC.
import numpy as np
import astropy.constants as c
import astropy.units as u
from astropy.io import ascii
from scipy.integrate import quad
import emcee
import corner
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

data = ascii.read('apj341188t13_mrt.txt', format='mrt')
z, mB, mB_err, x1, x1_err, cc, c_err = data['zCMB'], data['Bmag'], data['e_Bmag'], data['x1'], data['e_x1'], data['c'], data['e_c']


def Dl(z, H0, OmegaM, OmegaDE, w):
    OmegaK = 1 - OmegaM - OmegaDE
    DH = (c.c / (H0 * u.km/u.s/u.Mpc)).to(u.Mpc).value

    def E(z):
        return np.sqrt(OmegaM*(1+z)**3 + OmegaK*(1+z)**2 + OmegaDE*(1+z)**(3*(1+w)))
    
    I = np.array([quad(lambda z: 1/E(z), 0, zi)[0] for zi in z])

    if OmegaK == 0:
        d = I * DH * (1 + z) 
    elif OmegaK > 0:
        d = DH * np.sinh(np.sqrt(OmegaK) * I) * (1 + z) / np.sqrt(OmegaK)
    else:
        d = DH * np.sin(np.sqrt(-OmegaK) * I) * (1 + z) / np.sqrt(-OmegaK)

    d[d<0] = 1e-10
    return d

def dist_modulus(z, H0, OmegaM, OmegaDE, w):
    return 5 * np.log10(Dl(z, H0, OmegaM, OmegaDE, w)) + 25


def muB(mB, x1, c, alpha, beta, M):
    return mB - M + alpha * x1 - beta * c


def muB_err(mB_err, x1_err, c_err, alpha, beta):
    return np.sqrt(mB_err**2 + alpha**2 * x1_err**2 + beta**2 * c_err**2)


def lnprob(p):
    H0, OmegaM, OmegaDE, w, alpha, beta, M = p
    mu = muB(mB, x1, cc, alpha, beta, M)
    mu_err = muB_err(mB_err, x1_err, c_err, alpha, beta)
    if np.isnan(-0.5 * np.sum((mu - dist_modulus(z, H0, OmegaM, OmegaDE, w))**2 / mu_err**2)):
        return print(p)
    return -0.5 * np.sum((mu - dist_modulus(z, H0, OmegaM, OmegaDE, w))**2 / mu_err**2)


if __name__ == '__main__':
    np.random.seed(42)

    params = ['H0', 'OmegaM', 'OmegaDE', 'w', 'alpha', 'beta', 'M']
    nwalkers = 32
    ndim = len(params)
    p0 = np.random.normal(1, 0.1, (nwalkers, ndim)) * np.array([70, 0.3, 0.7, -1, 0.112, 3.1, -19])

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        start = time.time()
        sampler.run_mcmc(p0, 5000, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} mins".format(multi_time/60))


    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    corner.corner(samples, labels=params)
    # save samples
    np.save('samples.npy', samples)
    
    # save corner plot  
    plt.savefig('corner_plot.png')
