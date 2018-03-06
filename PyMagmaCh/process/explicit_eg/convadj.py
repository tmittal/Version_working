import numpy as np
from climlab import constants as const
from climlab.process.time_dependent_process import TimeDependentProcess
from climlab.domain.field import Field


class ConvectiveAdjustment(TimeDependentProcess):
    '''Convective adjustment process
    Instantly returns column to neutral lapse rate

    Adjustment includes the surface IF 'Ts' is included in the state
    dictionary. Otherwise only the atmopsheric temperature is adjusted.'''
    def __init__(self, adj_lapse_rate=None, **kwargs):
        super(ConvectiveAdjustment, self).__init__(**kwargs)
        self.param['adj_lapse_rate'] = adj_lapse_rate
        self.time_type = 'adjustment'
        self.adjustment = {}

    def compute(self):
        #lapse_rate = self.param['adj_lapse_rate']
        Tadj = convective_adjustment_direct(self.pnew, Tcol, self.cnew, lapserate=self.adj_lapse_rate)
        Tatm = Field(Tadj[...,1:], domain=self.Tatm.domain)
        self.adjustment['Ts'] = Ts - self.Ts
        self.adjustment['Tatm'] = Tatm - self.Tatm


# @jit  # numba.jit not working here. Not clear why.
#  At least we get something like 10x speedup from the inner loop
#  Attempt to use numba to compile the Akamaev_adjustment function
#  which gives at least 10x speedup
#   If numba is not available or compilation fails, the code will be executed
#   in pure Python. Results should be identical
try:
    from numba import jit
    Akamaev_adjustment = jit(signature_or_function=Akamaev_adjustment)
    #print 'Compiling Akamaev_adjustment() with numba.'
except:
    pass
