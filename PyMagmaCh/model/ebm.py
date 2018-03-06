"""Base Model -
 This is the file that defines the model parameters mostly
 defined in flood_basalts_v8.m file.

 Object-oriented code for a coupled magma chamber model

 Requirements for the model :
 a. Allow an arbitrary number of chambers
 b. Allow specifying most of the parameters (can give default)
 c. Allow specifying which processes to turn on and off
    (modularity is important)

Code developed by Ben Black and Tushar Mittal
"""

import numpy as np
from machlab import constants as const
from climlab.surface import albedo

from climlab.domain.field import Field, global_mean
from climlab.domain import domain
from climlab.radiation.AplusBT import AplusBT
from climlab.radiation.insolation import P2Insolation, AnnualMeanInsolation, DailyInsolation
from climlab.dynamics.diffusion import MeridionalDiffusion
from climlab.process.energy_budget import EnergyBudget

from scipy import integrate

class EBM(EnergyBudget):
    def __init__(self,
                 num_lat=90,
                 S0=const.S0,
                 A=210.,
                 B=2.,
                 D=0.555,  # in W / m^2 / degC, same as B
                 water_depth=10.0,
                 Tf=-10.,
                 a0=0.3,
                 a2=0.078,
                 ai=0.62,
                 timestep=const.seconds_per_year/90.,
                 T_init_0 = 12.,
                 T_init_P2 = -40.,
                 **kwargs):
        super(EBM, self).__init__(timestep=timestep, **kwargs)
        if not self.domains and not self.state:  # no state vars or domains yet
            sfc = domain.zonal_mean_surface(num_lat=num_lat,
                                            water_depth=water_depth)
            lat = sfc.axes['lat'].points
            initial = T_init_0 + T_init_P2 * legendre.P2(np.sin(np.deg2rad(lat)))
            self.set_state('Ts', Field(initial, domain=sfc))
        self.param['S0'] = S0
        self.param['A'] = A
        self.param['B'] = B
        self.param['D'] = D
        self.param['Tf'] = Tf
        self.param['water_depth'] = water_depth
        self.param['a0'] = a0
        self.param['a2'] = a2
        self.param['ai'] = ai
        # create sub-models
        self.add_subprocess('LW', AplusBT(state=self.state, **self.param))
        self.add_subprocess('insolation',
                            P2Insolation(domains=sfc, **self.param))
        self.add_subprocess('albedo',
                            albedo.StepFunctionAlbedo(state=self.state,
                                                      **self.param))
        # diffusivity in units of 1/s
        K = self.param['D'] / self.domains['Ts'].heat_capacity
        self.add_subprocess('diffusion', MeridionalDiffusion(state=self.state,
                                                             K=K,
                                                             **self.param))
        self.topdown = False  # call subprocess compute methods first

    def _compute_heating_rates(self):
        '''Compute energy flux convergences to get heating rates in W / m**2.
        This method should be over-ridden by daughter classes.'''
        insolation = self.subprocess['insolation'].diagnostics['insolation']
        albedo = self.subprocess['albedo'].diagnostics['albedo']
        ASR = (1-albedo) * insolation
        self.heating_rate['Ts'] = ASR
        self.diagnostics['ASR'] = ASR
        self.diagnostics['net_radiation'] = (ASR -
                                    self.subprocess['LW'].diagnostics['OLR'])

    def global_mean_temperature(self):
        '''Convenience method to compute global mean surface temperature.'''
        return global_mean(self.state['Ts'])

    def inferred_heat_transport(self):
        '''Returns the inferred heat transport (in PW)
        by integrating the TOA energy imbalance from pole to pole.'''
        phi = np.deg2rad(self.lat)
        energy_in = np.squeeze(self.diagnostics['net_radiation'])
        return (1E-15 * 2 * np.math.pi * const.a**2 *
                integrate.cumtrapz(np.cos(phi)*energy_in, x=phi, initial=0.))

    def heat_transport(self):
        '''Returns instantaneous heat transport in units on PW,
        on the staggered grid.'''
        return self.diffusive_heat_transport()

    def diffusive_heat_transport(self):
        '''Compute instantaneous diffusive heat transport in units of PW
        on the staggered grid.'''
        phi = np.deg2rad(self.lat)
        phi_stag = np.deg2rad(self.lat_bounds)
        D = self.param['D']
        T = np.squeeze(self.Ts)
        dTdphi = np.diff(T) / np.diff(phi)
        dTdphi = np.append(dTdphi, 0.)
        dTdphi = np.insert(dTdphi, 0, 0.)
        return (1E-15*-2*np.math.pi*np.cos(phi_stag)*const.a**2*D*dTdphi)

    def heat_transport_convergence(self):
        '''Returns instantaneous convergence of heat transport
        in units of W / m^2.'''
        phi = np.deg2rad(self.lat)
        phi_stag = np.deg2rad(self.lat_bounds)
        H = 1.E15*self.heat_transport()
        return (-1./(2*np.math.pi*const.a**2*np.cos(phi)) *
                np.diff(H)/np.diff(phi_stag))


class EBM_seasonal(EBM):
    def __init__(self, a0=0.33, a2=0.25, ai=None, **kwargs):
        '''This EBM uses realistic daily insolation.
        If ai is not given, the model will not have an albedo feedback.'''
        super(EBM_seasonal, self).__init__(a0=a0, a2=a2, ai=ai, **kwargs)
        sfc = self.domains['Ts']
        self.add_subprocess('insolation',
                            DailyInsolation(domains=sfc, **self.param))
        self.param['a0'] = a0
        self.param['a2'] = a2
        if ai is None:
            # No albedo feedback
            # Remove unused parameters here for clarity
            _ = self.param.pop('ai')
            _ = self.param.pop('Tf')
            self.add_subprocess('albedo',
                            albedo.P2Albedo(domains=sfc, **self.param))
        else:
            self.param['ai'] = ai
            self.add_subprocess('albedo',
                    albedo.StepFunctionAlbedo(state=self.state, **self.param))



#==============================================================================
# class EBM_landocean( EBM_seasonal ):
#     '''A model with both land and ocean, based on North and Coakley (1979)
#     Essentially just invokes two different EBM_seasonal objects, one for ocean, one for land.
#     '''
#     def __str__(self):
#         return ( "Instance of EBM_landocean class with " +  str(self.num_points) + " latitude points." )
#
#     def __init__( self, num_points = 90 ):
#         super(EBM_landocean,self).__init__( num_points )
#         self.land_ocean_exchange_parameter = 1.0  # in W/m2/K
#
#         self.land = EBM_seasonal( num_points )
#         self.land.make_insolation_array( self.orb )
#         self.land.Tf = 0.
#         self.land.set_timestep( timestep = self.timestep )
#         self.land.set_water_depth( water_depth = 2. )
#
#         self.ocean = EBM_seasonal( num_points )
#         self.ocean.make_insolation_array( self.orb )
#         self.ocean.Tf = -2.
#         self.ocean.set_timestep( timestep = self.timestep )
#         self.ocean.set_water_depth( water_depth = 75. )
#
#         self.land_fraction = 0.3 * np.ones_like( self.land.phi )
#         self.C_ratio = self.land.water_depth / self.ocean.water_depth
#         self.T = self.zonal_mean_temperature()
#
#     def zonal_mean_temperature( self ):
#         return self.land.T * self.land_fraction + self.ocean.T * (1-self.land_fraction)
#
#     def step_forward( self ):
#         #  note.. this simple implementation is possibly problematic
#         # because the exchange should really occur simultaneously with radiation
#         # and before the implicit heat diffusion
#         self.exchange = (self.ocean.T - self.land.T) * self.land_ocean_exchange_parameter
#         self.land.step_forward()
#         self.ocean.step_forward()
#         self.land.T += self.exchange / self.land_fraction * self.land.delta_time_over_C
#         self.ocean.T -= self.exchange / (1-self.land_fraction) * self.ocean.delta_time_over_C
#         self.T = self.zonal_mean_temperature()
#         self.update_time()
#
#     #   This code should be more accurate, but it's ungainly and seems to produce just about the same result.
#     #def step_forward( self ):
#     #    self.exchange = (self.ocean.T - self.land.T) * self.land_ocean_exchange_parameter
#     #    self.land.compute_radiation( )
#     #    self.ocean.compute_radiation( )
#     #    Trad_land = ( self.land.T + ( self.land.net_radiation + self.exchange / self.land_fraction )
#     #        * self.land.delta_time_over_C )
#     #    Trad_ocean = ( self.ocean.T + ( self.ocean.net_radiation - self.exchange / (1-self.land_fraction) )
#     #        * self.ocean.delta_time_over_C )
#     #    self.land.T = solve_banded((1,1), self.land.diffTriDiag, Trad_land )
#     #    self.ocean.T = solve_banded((1,1), self.ocean.diffTriDiag, Trad_ocean )
#     #    self.T = self.zonal_mean_temperature()
#     #    self.land.update_time()
#     #    self.ocean.update_time()
#     #    self.update_time()
#
#     def integrate_years(self, years=1.0, verbose=True ):
#         #  Here we make sure that both sub-models have the current insolation.
#         self.land.make_insolation_array( self.orb )
#         self.ocean.make_insolation_array( self.orb )
#        super(EBM_landocean,self).integrate_years( years, verbose )
#==============================================================================
