from scipy.optimize import root

import numpy as np
from SMT.core.physical_constants import *

def Fermi_Energy(ne):
    E_F = 1/(2*m_e) * (3*π**2 * ne)**(2/3)
    return E_F

def Fermi_velocity(ne):
    v_F = np.sqrt(2*Fermi_Energy(ne)/m_e)
    return v_F

def Fermi_wavenumber(ne):
    k_F = Fermi_velocity(ne)*m_e
    return k_F

def Degeneracy_Parameter(Te, ne):
    θ = Te/Fermi_Energy(ne)
    return θ

def Ion_Plasma_Frequency(ni, mi, Zbar):
    ωp = np.sqrt(4 * π *Zbar**2 * ni /mi)
    return ωp

def Electron_Plasma_Frequency(ne):
    ωe = np.sqrt(4 * π * ne /m_e)
    return ωe

def Gamma(T, n, Z):
    β = 1/T
    rs = rs_from_n(n)
    return Z**2*β/rs

def Debye_length(T, ni, Zbar):
    ne = Zbar*ni
    # λD = 1/np.sqrt(  4*π*ne/T + 4*π*Zbar**2*ni/T  )
    λD = 1/np.sqrt(  4*π*ne/T )
    return λD

def Thomas_Fermi_screening_length(T, ni, Zbar):
    ne = Zbar*ni
    E_F = Fermi_Energy(ne)
    # λTF = 1/np.sqrt(  4*π*ne/np.sqrt(T**2 + (2/3*E_F)**2) )
    λTF = 1/np.sqrt(  4*π*ne/(T**1.8 + (2/3*E_F)**1.8)**(5/9) )
    return λTF

def thermal_deBroglie_wavelength(T, m):
    # return np.sqrt(2*π/ (m*T) )
    return 1/np.sqrt(2*π*m*T)

def Kappa(T, ni, Zbar):
    ri = rs_from_n(ni)
    λD = Thomas_Fermi_screening_length(T, ni, Zbar)
    return ri/λD

def n_from_rs( rs):
    """
    Sphere radius to density, in any units
    """
    return 1/(4/3*π*rs**3)

def rs_from_n(n):
    """
    Density to sphere radius, in any units.
    """
    return (4/3*π*n)**(-1/3)

def ThomasFermiZbar( Z, n_AU, T_AU):
        """
        Finite Temperature Thomas Fermi Charge State using 
        R.M. More, "Pressure Ionization, Resonances, and the
        Continuity of Bound and Free States", Adv. in atomic 
        Mol. Phys., Vol. 21, p. 332 (Table IV).

        Z = atomic number
        n_AU = number density AU
        T = temperature AU
        """

        alpha = 14.3139
        beta = 0.6624
        a1 = 0.003323
        a2 = 0.9718
        a3 = 9.26148e-5
        a4 = 3.10165
        b0 = -1.7630
        b1 = 1.43175
        b2 = 0.31546
        c1 = -0.366667
        c2 = 0.983333

        n_cc = n_AU * AU_to_invcc
        T_eV = T_AU * AU_to_eV

        convert = n_cc*1.6726219e-24
        R = convert/Z
        T0 = T_eV/Z**(4./3.)
        Tf = T0/(1 + T0)
        A = a1*T0**a2 + a3*T0**a4
        B = -np.exp(b0 + b1*Tf + b2*Tf**7)
        C = c1*Tf + c2
        Q1 = A*R**B
        Q = (R**C + Q1**C)**(1/C)
        x = alpha*Q**beta

        return Z*x/(1 + x + np.sqrt(1 + 2.*x))

# XC fits
def xc_PDW_h(θ): 
    """
    See [4]
    """
    N = 1 + 2.8343*θ**2 - 0.2151*θ**3 + 5.2759*θ**4
    D = 1 + 3.9431*θ**2 + 7.9138*θ**4
    h = N/D * np.tanh(1/θ)
    return h

# XC fits
def xc_YOT(Te, ne): 
    """
    See [5]
    """
    θ = Degeneracy_Parameter(Te, ne)
    β = 1/Te
    sech = lambda x: 1/np.cosh(x)
    γx = 1/(2*π) * (3*ne/(8*π))**(1/3) * np.tanh(4/(9*θ)) + ne/(12*Te) * sech(4/(9*θ))**2
    a = -1.51
    b = -0.840
    c = 0.275
    d = -0.553

    γc = 1/4 * (β*ne/π)**(1/2) *( np.tanh(c*β**a*ne**b/(1+β**d)) + 2/3 * b*c*β**a/(1 + β**d*ne**(b+1/2))*sech(c*β**a*ne**b/(1+β**d))**2  )
    γxc = γx + γc
    return γxc

