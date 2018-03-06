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
from PyMagmaCh import constants as const
from PyMagmaCh.domain.field import Field
from PyMagmaCh.domain import domain
from PyMagmaCh.process.time_dependent_process import TimeDependentProcess


from PyMagmaCh.radiation.insolation import FixedInsolation
from PyMagmaCh.radiation.radiation import Radiation, RadiationSW
from PyMagmaCh.convection.convadj import ConvectiveAdjustment
from PyMagmaCh.surface.surface_radiation import SurfaceRadiation
from PyMagmaCh.radiation.nband import ThreeBandSW, FourBandLW, FourBandSW
from PyMagmaCh.radiation.water_vapor import ManabeWaterVapor

class MagmaChamberModel(TimeDependentProcess):
    def __init__(self,
                 num_depth=30,
                 num_rad=1,
                 depth=None,
                 radial_val=None,
                 abs_coeff = 1.,
                 timestep=1.* const.seconds_per_year,
                 **kwargs):
        if depth is not None:
            num_depth = np.array(depth).size
        if radial_val is not None:
            num_rad = np.array(radial_val).size
        # Check to see if an initial state is already provided
        #  If not, make one
        if 'state' not in kwargs:
            state = self.initial_state(num_depth, num_rad, depth,radial_val)
            kwargs.update({'state': state})
        super(MagmaChamberModel, self).__init__(timestep=timestep, **kwargs)
        self.param['abs_coeff'] = abs_coeff
        #z_clmn, atm_slab
        z_clmn = self.Tdepth.domain
        atm_slab = self.Ts.domain
        # create sub-models for longwave and shortwave radiation
        dp = self.Tdepth.domain.lev.delta
        absorbLW = compute_layer_absorptivity(self.param['abs_coeff'], dp)
        absorbLW = Field(np.tile(absorbLW, z_clmn.shape), domain=z_clmn)
        absorbSW = np.zeros_like(absorbLW)
        longwave = Radiation(state=self.state, absorptivity=absorbLW,
                             albedo_z_clmn=0)
        shortwave = RadiationSW(state=self.state, absorptivity=absorbSW,
                                albedo_z_clmn=self.param['albedo_z_clmn'])
        # sub-model for insolation ... here we just set constant Q
        thisQ = self.param['Q']*np.ones_like(self.Ts)
        Q = FixedInsolation(S0=thisQ, domain=z_clmn, **self.param)
        #  surface sub-model
        surface = SurfaceRadiation(state=self.state, **self.param)
        self.add_subprocess('LW', longwave)
        self.add_subprocess('SW', shortwave)
        self.add_subprocess('insolation', Q)
        self.add_subprocess('surface', surface)

    def initial_state(self, num_lev, num_lat, lev, lat, water_depth):
        return initial_state(num_lev, num_lat, lev, lat, water_depth)

    # This process has to handle the coupling between insolation and column radiation
    def compute(self):
        # some handy nicknames for subprocesses
        LW = self.subprocess['LW']
        SW = self.subprocess['SW']
        insol = self.subprocess['insolation']
        surf = self.subprocess['surface']
        # Do the coupling
        SW.flux_from_space = insol.diagnostics['insolation']
        SW.albedo_z_clmn = surf.albedo_z_clmn
        surf.LW_from_atm = LW.flux_to_z_clmn
        surf.SW_from_atm = SW.flux_to_z_clmn
        LW.flux_from_z_clmn = surf.LW_to_atm
        # set diagnostics
        self.do_diagnostics()

    def do_diagnostics(self):
        '''Set all the diagnostics from long and shortwave radiation.'''
        LW = self.subprocess['LW']
        SW = self.subprocess['SW']
        surf = self.subprocess['surface']
        try: self.diagnostics['OLR'] = LW.flux_to_space
        except: pass
        try: self.diagnostics['LW_down_z_clmn'] = LW.flux_to_z_clmn
        except: pass
        try: self.diagnostics['LW_up_z_clmn'] = surf.LW_to_atm
        except: pass
        try: self.diagnostics['LW_absorbed_z_clmn'] = (surf.LW_from_atm -
                                                    surf.LW_to_atm)
        except: pass
        try: self.diagnostics['LW_absorbed_atm'] = LW.absorbed
        except: pass
        try: self.diagnostics['LW_emission'] = LW.emission
        except: pass
            #  contributions to OLR from surface and atm. levels
            #self.diagnostics['OLR_z_clmn'] = self.flux['z_clmn2space']
            #self.diagnostics['OLR_atm'] = self.flux['atm2space']
        try: self.diagnostics['ASR'] = SW.flux_from_space - SW.flux_to_space
        except: pass
        try:
            self.diagnostics['SW_absorbed_z_clmn'] = (surf.SW_from_atm -
                                                    surf.SW_to_atm)
        except: pass
        try: self.diagnostics['SW_absorbed_atm'] = SW.absorbed
        except: pass
        try: self.diagnostics['SW_down_z_clmn'] = SW.flux_to_z_clmn
        except: pass
        try: self.diagnostics['SW_up_z_clmn'] = SW.flux_from_z_clmn
        except: pass
        try: self.diagnostics['SW_up_TOA'] = SW.flux_to_space
        except: pass
        try: self.diagnostics['SW_down_TOA'] = SW.flux_from_space
        except: pass
        try: self.diagnostics['SW_absorbed_total'] = (SW.absorbed_total -
                                                      SW.flux_net[0])
        except: pass
        try: self.diagnostics['planetary_albedo'] = (SW.flux_to_space /
                                                     SW.flux_from_space)
        except: pass
        try: self.diagnostics['SW_emission'] = SW.emission
        except: pass

######### Need to fix the intial temperature field
def initial_state(num_depth, num_rad, depth,radial_val,geotherm_grad):
    if num_rad is 1:
        z_clmn, atm_slab = domain.z_column(num_depth=num_depth,depth=depth)
    else:
        z_clmn, atm_slab = domain.z_radial_column(num_depth=num_depth, num_rad=num_rad,
                                            depth=depth,
                                            radial_val = radial_val)
    num_dpth = z_clmn.depth.num_points
    Ts = Field(const.surface_temp*np.ones(atm_slab.shape), domain=atm_slab)
    Tinitial = np.tile(np.linspace(288., 1000, num_dpth), atm_slab.shape) # const.geotherm_grad
    Tdepth = Field(Tinitial, domain=z_clmn)
    state = {'Ts': Ts, 'Tdepth': Tdepth}
    return state


class RadiativeConvectiveModel(GreyRadiationModel):
    def __init__(self,
                 # lapse rate for convective adjustment, in K / km
                 adj_lapse_rate=6.5,
                 **kwargs):
        super(RadiativeConvectiveModel, self).__init__(**kwargs)
        self.param['adj_lapse_rate'] = adj_lapse_rate
        self.add_subprocess('convective adjustment', \
            ConvectiveAdjustment(state=self.state, **self.param))
