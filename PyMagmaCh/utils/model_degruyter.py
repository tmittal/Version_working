'''model_degruyer.py

A collection of function definitions to handle common
calcualtions (i.e. constants and melting curve, density parameterizations)
in Degruyter & Huber 2014 paper

'''
import numpy as np
from PyMagmaCh.utils import constants as const

crust_density = 2600. # kg/m^3

def gas_density_degruyter(T,P):
    """Compute equation of state of the gas phase.

    Input:  T is temperature in Kelvin ( 873 < T < 1173 K )
            P is pressure in Pa (30 Mpa < P < 400 MPa)
    Output: rhog, drhog_dP, drhog_dT (gas density, d(rho_g)/dP and d(rho_g)/dT)
    """
    rhog = 1e3*(-112.528*(T**(-0.381)) + 127.811*(P**(-1.135)) + 112.04*(T**(-0.411))*(P**(0.033))) # Units :kg/m^3
    drhog_dT = 1e3*((-0.381)*-112.528*(T**(-1.381)) + (-0.411)*112.04*(T**(-1.411))*(P**(0.033)))
    drhog_dP = 1e-2*((-1.135)*127.811*(P**(-2.135)) + (0.033)*112.04*(T**(-0.411))*(P**(-.9670)))
    return rhog, drhog_dP,drhog_dT

def melting_curve_degruyter(T,eta_g,b = 0.5,T_s=973.0,T_l=1223.0):
    """Compute melt fraction-temperature relationship.

    Input:  T is temperature in Kelvin
            eta_g is gas volume fraction
            b is an exponent to approximate composition (1 = mafic, 0.5 = silicic)
            T_s is solidus temperature in Kelvin (Default value = 973 K)
            T_l is liquidus temperature in Kelvin (Default value = 1223 K)
    Output: eta_x,deta_x_dT,deta_x_deta_g (eta_x is crystal volume fraction, others are its derivative with T and eta_g)
    """
    temp1 = T - T_s
    temp2 = T_l - T_s
    eta_x = (1. - eta_g)*(1. - (temp1/temp2)**b)
    deta_x_dT = (1. - eta_g)*(-b*(temp1)**(b-1.)/(temp2)**b)
    deta_x_deta_g = -1.*(1. - (temp1/temp2)**b)
    return eta_x,deta_x_dT,deta_x_deta_g

def solubulity_curve_degruyter(T,P):
    """Compute solubility - dissolved water content in the melt

    Input:  T is temperature in Kelvin ( 873 < T < 1173 K )
            P is pressure in Pa (30 Mpa < P < 400 MPa)
    Output: meq,dmeq_dT,dmeq_dP (meq is dissolved water content others are its derivative with T and eta_g)
    """
    meq = 1e-2*(np.sqrt(P)*(0.4874 - 608./T + 489530.0/T**2.)
          + P*(-0.06062 + 135.6/T - 69200.0/T**2.)
          + (P**(1.5))*(0.00253 - 4.154/T + 1509.0/T**2.))  # is dimensionless
    dmeq_dP = 1e-8*(0.5*(P**(-0.5))*(0.4874 - 608./T + 489530.0/T**2.)
          + (-0.06062 + 135.6/T - 69200.0/T**2.)
          + 1.5*(P**(0.5))*(0.00253 - 4.154/T + 1509.0/T**2.))
    dmeq_dT = 1e-2*(np.sqrt(P)*(608./T**2.-2*489530.0/T**3.)
          + P*(-135.6/T**2. + 2.*69200.0/T**3.)
          + (P**(1.5))*(4.154/T**2. -2.*1509.0/T**3.))
    return meq,dmeq_dT,dmeq_deta_g


def crit_outflow_degruyter():
    """
    Specify the conditions for eruptions according to Degruyter 2014 model
    Pc = critical overpressure
    eta_x = crystal volume fraction
    M_out_rate is the mass outflow rate
    """
    delta_Pc = np.randint(10,50) # assume a critical overpressure btw 10 - 50 MPa
    eta_x = 0.5 # based on cystal locking above 50 % packing ..
    M_out_rate = 1e4 # kg/s
    return delta_Pc,eta_x,M_out_rate

def material_constants_degruyter():
    """
    Specify the material constants used in the paper -
    Output as a dictionary ..
    alpha_m = melt thermal expansion coefficient (1/K)
    alpha_x = crystal thermal expansion coefficient (1/K)
    alpha_r = crust thermal expansion coefficient (1/K)
    beta_x = melt bulk modulus (Pa)
    beta_m = crystal bulk modulus (Pa)
    beta_r = crust bulk modulus (Pa)
    k_crust = thermal conductivity of the crust (J/s/m/K)
    c_x,c_g,c_m = specific heat capacities (J/kg/K)
    L_m,L_e = latent heat of melting and exsolution (J/kg)
    kappa = thermal diffusivity of the crust
    """
    mat_const = {'beta_m': 1e10, 'alpha_m': 1e-5, 'beta_x': 1e10, 'alpha_x':1e-5, 'beta_r': 1e10, 'alpha_r':1e-5,
                'k_crust': 3.25,'c_m' : 1200.0,'c_X' : 1200.0,'c_g' : 3880.0,'L_m':27e4,'L_e':226e4,'kappa':1e-6}
    return mat_const


def crustal_viscosity_degruyter(T,p):
    """Compute the viscosity of the visco-elastic shell surrounding the magma chamber.

    Input:  T is temperature in Kelvin
            P
    Output:
    """

    theta = T*(const.ps/p)**const.kappa

def crustal_temp_radial_degruyter(R_0,S_scale,T_R,T_s=500.0,kappa=1e-6):
    """Analytical solution for the heat conduction equation - heat loss from magam chamber to the surrounding
        - Modeled as chamber being spherical (radius R_0)
          and the curstal section being a larger enclosed sphere (radius S)
        - Assumptions : Chmaber is isothermal (Chamber temp T = T_R)
    Input:  T_R  is temperature at the edge of the chamber (Kelvin)
            T_S  is temperature at the outer boundary of the visco-elastic shell (Kelvin, 500 K)
            kappa is thermal diffusivity of the crust (m^2/s, default = 1e-6 m^2/s)
    Output: dT_dR,eta_crust - temp gradient at chamber edge (R_0) and the averaged crustal viscosity
    dT_dR,eta_crust =  crustal_temp_model(R_0,self.diagnostics['S_scale'],
                                                      X[T_val],T_s = self.diagnostics['T_S']),
                                                      kappa = self.param['kappa'])
    """
    T_R0  = (R_0*T_R*(S - r) + S*T_S*(r- R_0))/r/(S - R_0) # initial temp T(r,t = 0)
    return dT_dR,eta_crust
