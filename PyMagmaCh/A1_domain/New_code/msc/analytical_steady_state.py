import numpy as np

def crustal_temp_radial_degruyter(r,t,T_R,kappa,R_0,S,T_s=500.0,kappa=1e-6):
    """Analytical solution for the heat conduction equation - heat loss from magam chamber to the surrounding
        - Modeled as chamber being spherical (radius R_0, units : m)
          and the curstal section being a larger enclosed sphere (radius S >> R0)
        - Assumptions : Chmaber is isothermal (Chamber temp T = T_R)
    Input:  T_R  is temperature at the edge of the chamber (Kelvin)
            T_S  is temperature at the outer boundary of the visco-elastic shell (Kelvin, 500 K)
            kappa is thermal diffusivity of the crust (m^2/s, default = 1e-6 m^2/s)
            Note that S should be same once chosen ... (since it sets the inital bkg temp gradient etc ..)
    Output: .
    """
    T_R0  = (R_0*T_R*(S - r) + S*T_S*(r- R_0))/r/(S - R_0) # initial temp T(r,t = 0)
    delta_ch = S - R_0
    tmp1 = 2.*np.pi*kappa*R_0/r/delta_ch**2.
    for n in range(50): # truncate the series after first 50 terms ..
        tmp2 = n*np.sin(n*np.pi*(r - R_0)/delta_ch)
        tmp5 = 1. - tmp3
    return theta
