
import numpy as np
import astropy.constants as c
import astropy.units as u
from astropy.io import ascii
from scipy.integrate import quad

def mag(d, M=-19.5):
    """
    Convert distance(s) `d` into magnitude(s) given 
    absolute magnitude `M`. Assumes `d` is in units of Mpc.
    
    """
    return M + 5. * np.log10(d * 1e6)

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


def cosmocalc(z, Omega_m=0.3, Omega_L=0.7, Omega_r=0.0, H0=70., N=1000):
    Omega_K = 1. - Omega_m - Omega_L
    Tyr = (1/(u.km/u.s/u.Mpc)).to('Gyr').value  # Conversion from 1/H to Gyr
    az = 1.0 / (1. + z)  # Scale factor at redshift z
    h = H0 / 100.  # Normalized H0
    cc = c.c.to('km/s').value  # Speed of light in km/s

    zage, DTT, DCMR = 0., 0., 0.
    for i in range(N):
        a = az * (i + 0.5) / N
        adot = np.sqrt(Omega_K + (Omega_m / a) + (Omega_r / a**2) + (Omega_L * a**2))
        zage += 1. / adot

        a = az + (1 - az) * (i + 0.5) / N
        adot = np.sqrt(Omega_K + (Omega_m / a) + (Omega_r / a**2) + (Omega_L * a**2))
        DTT += 1. / adot
        DCMR += 1. / (a * adot)

    # Convert age and distances to more useful units
    zage_Gyr = zage * Tyr * az / N / H0
    DTT_Gyr = DTT * Tyr * (1. - az) / N / H0
    DCMR_Mpc = cc / H0 * DCMR * (1. - az) / N

    # Compute angular diameter distance and luminosity distance
    DA_Mpc = az * DCMR_Mpc
    DL_Mpc = DA_Mpc / (az * az)

    # Comoving volume calculation
    x = np.sqrt(abs(Omega_K)) * DCMR
    if x > 0.1:
        ratio = np.sinh(x) / x if Omega_K > 0 else np.sin(x) / x
    else:
        y = x**2
        ratio = 1. + y / 6. + y**2 / 120.

    VCM = ratio * DCMR**3 / 3
    V_Gpc = 4. * np.pi * (1e-3 * cc / H0)**3 * VCM  # convert to Gpc

    return zage_Gyr, DCMR_Mpc, DA_Mpc, DL_Mpc, V_Gpc
