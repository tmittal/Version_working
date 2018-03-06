'''constants.py

A collection of physical constants for the magma chamber models.

Part of the machlab package
Ben Black, Tushar Mittal
'''

import numpy as np

a_earth = 6367444.7          # Radius of Earth (m)
a_mars  = 3386000.0          # Radius of Mars (m)
g_earth = 9.807          # gravitational acceleration Earth (m / s**2)
g_mars  = 3.711          # gravitational acceleration Mars  (m / s**2)

#  Some useful time conversion factors
seconds_per_minute = 60.
minutes_per_hour = 60.
hours_per_day = 24.

# the length of the "tropical year" -- time between vernal equinoxes
days_per_year = 365.2422
seconds_per_hour = minutes_per_hour * seconds_per_minute
minutes_per_day = hours_per_day * minutes_per_hour
seconds_per_day = hours_per_day * seconds_per_hour
seconds_per_year = seconds_per_day * days_per_year
minutes_per_year = seconds_per_year / seconds_per_minute
hours_per_year = seconds_per_year / seconds_per_hour
#  average lenghts of months based on dividing the year into 12 equal parts
months_per_year = 12.
seconds_per_month = seconds_per_year / months_per_year
minutes_per_month = minutes_per_year / months_per_year
hours_per_month = hours_per_year / months_per_year
days_per_month = days_per_year / months_per_year

######################################################
# Some parameter values
######################################################

rho_water = 1000.      # density of water (kg / m**3)
cp_water = 4181.3      # specific heat of liquid water (J / kg / K)

tempCtoK = 273.15   # 0degC in Kelvin
tempKtoC = -tempCtoK  # 0 K in degC
bar_to_Pa = 1e5  # conversion factor from bar to Pa

kBoltzmann = 1.3806488E-23  # the Boltzmann constant (J / K)
c_light = 2.99792458E8   # speed of light (m/s)
hPlanck = 6.62606957E-34  # Planck's constant (J s)
# Stef_Boltz_sigma = 5.67E-8  # Stefan-Boltzmann constant (W / m**2 / K**4)
# Stef_Boltz_sigma derived from fundamental constants
Stef_Boltz_sigma = (2*np.pi**5 * kBoltzmann**4) / (15 * c_light**2 * hPlanck**3)

######################################################

crys_frac_lock_Marsh = 0.55  # critical crystal fraction for crystal locking
                             # (Marsh 2015, Treatise of Geophysics Vol2)

D_h2o_Katz2003 = 0.01;   # constant from Katz, 2003 hydrous melting model
D_co2_Katz2003 = 0.0001; # CO2 is highly incompatible;
                         # cf. E.H. Hauri et al. / EPSL 248 (2006) 715?734
                         # Used to calculate the conc. of water and CO2 in melt
                         # for given bulk composition and degree of melting
##########################################################
# Some constants for the code - default values ...

moho_depth = 30.*1e3         # Set default extent of the depth domain to be 30 km
region_size = 100.*1e3  # Set default extent of the x_val domain to be 100 km
surface_temp = 288. # Surface temperature is fixed to 288 Kelvin
geotherm_grad = 30. # Kelvin/km, typical geotherm_gradient



######################################################
# Things to add : Have some standard parameter values for things like
# basalts, ultramafic melts, pyrolitic, garnet etc
# have latent heat, density, viscosity model, composition

Lhvap = 2.5E6    # Latent heat of vaporization (J / kg)
Lhsub = 2.834E6   # Latent heat of sublimation (J / kg)
Lhfus = Lhsub - Lhvap  # Latent heat of fusion (J / kg)
cp = 1004.     # specific heat at constant pressure for dry air (J / kg / K)
Rd = 287.         # gas constant for dry air (J / kg / K)
kappa = Rd / cp
Rv = 461.5       # gas constant for water vapor (J / kg / K)

######################################################
# From Ben's model
cp_cc_Ben   =   0.8e3    # J/kg/K quartz
cp_mag_Ben  =   0.84e3  # J/kg/K basalt
k_Ben       =   1.5     # Thermal conductivity (W/m/C)
rhocrust_Ben = 3000.    # density kg/m3
