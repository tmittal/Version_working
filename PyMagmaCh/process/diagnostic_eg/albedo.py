import numpy as np
from PyMagmaCh.process.time_dependent_process import TimeDependentProcess
from PyMagmaCh.A1_domain.field import Field

class Iceline(TimeDependentProcess):
    def __init__(self, Tf=-10., **kwargs):
        super(DiagnosticProcess, self).__init__(**kwargs)
        self.time_type = 'diagnostic'
        self.param['Tf'] = Tf

    def find_icelines(self):
        Tf = 5.
        Ts = self.state['a1']
        #lat_bounds = self.domains['a1'].axes['lat'].bounds
        noice = np.where(Ts >= Tf, True, False)
        ice = np.where(Ts < Tf, True, False)
        self.diagnostics['noice'] = noice
        self.diagnostics['ice'] = ice
        if ice.all():
            # 100% ice cover
            icelat = np.array([-0., 0.])
        elif noice.all():
            # zero ice cover
            icelat = np.array([-90., 90.])
        else:  # there is some ice edge
            # Taking np.diff of a boolean array gives True at the boundaries between True and False
            boundary_indices = np.where(np.diff(ice.squeeze()))[0] + 1
            #icelat = lat_bounds[boundary_indices]  # an array of boundary latitudes
        #self.diagnostics['icelat'] = icelat


    def compute(self):
        self.find_icelines()


class StepFunctionAlbedo(TimeDependentProcess):
    def __init__(self, Tf=-10., a0=0.3, a2=0.078, ai=0.62, **kwargs):
        super(DiagnosticProcess, self).__init__(**kwargs)
        self.param['Tf'] = Tf
        self.param['a0'] = a0
        self.param['a2'] = a2
        self.param['ai'] = ai
        sfc = self.domains_var['a1']
        self.add_subprocess('iceline', Iceline(Tf=Tf, state=self.state))
        self.topdown = False  # i.e call subprocess compute methods first
        self.time_type = 'diagnostic'

    def _get_current_albedo(self):
        '''Simple step-function albedo based on ice line at temperature Tf.'''
        ice = self.subprocess['iceline'].diagnostics['ice']
        # noice = self.subprocess['iceline'].diagnostics['noice']
        albedo = Field(np.where(ice, 1., 0.), domain=self.domains_var['a1'])
        return albedo

    def compute(self):
        self.diagnostics['albedo'] = self._get_current_albedo()
