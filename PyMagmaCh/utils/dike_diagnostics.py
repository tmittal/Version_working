import numpy as np
from PyMagmaCh.utils import constants as const

def dike_diagnostics(rho_r =2800.,rho_f = 2500.,T_0 = 5e6) :
    '''Calculates Dike diagnostics given some inputs - Philipp, Afsar, & Gudmunsson Front Earth Sci 2013
    T_0 - tensile strength of the rock (in Pascal - so default is 5 MPa)

    '''
    #%%%%%%%%%%%%%%%%%%%%%%% Initialization%%%%%%%%%%%%%%%%%%%%%%%%%
    # Inputs needed (all units - meters, seconds, m/s, Pascal, Kelvin unless otherwise noted)-
    b = 1 # Dike aperture (across dimension width)
    W = 450. # Dike length (direction perpendicular to the fluid flow direction)
    mu_visc_fl = 1e-3 # Fluid viscosity
    alpha = np.pi/2. # dike dip in radians
    nu = 0.25 # poisson ratio
    E  = 10*1e9 # Young modulus of crustal rock (~ 5 -100 GPa)
    pe = 5.*1e6 # fluid excess pressure (~ 5-20 MPa)
    p_over  = (b/2./W)*E/(1. - nu**2.)
    h_dike = (p_over - pe)/((rho_r - rho_f)*const.g_earth)
    dpe_dD = -pe/h_dike
    Q_e = ((b**3.)*W/12./mu_visc_fl)*((rho_r - rho_f)*const.g_earth*np.sin(alpha) - dpe_dD)
    dike_prop ={'Q':Q_e,'p_over' : p_over/1e6,'pe':pe/1e6,'h_dike':h_dike/1e3} # pressure output in Mpa, dike_height in km
    return dike_prop

dike_diagnostics()
