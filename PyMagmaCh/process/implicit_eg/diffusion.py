import numpy as np
from scipy.linalg import solve_banded
from PyMagmaCh.process.process import Process


class Diffusion_1D(TimeDependentProcess):
    '''Parent class for implicit diffusion modules.
    Solves the 1D heat equation
    \rho C_p dT/dt = d/dx( K * dT/dx )

    The thermal conductivity K, density \rho and heat capacity Cp are in
    units  - W/m/K, kg/m^3, and J/K/kg.

    Assume that the boundary conditions are fixed temp ..
    (fix temp at the base and the top boundary ..) - need to specify
    self.param['T_base'],self.param['T_top'] :
    Note that the base and top are base and top of the grid

    Requirements : Can have only a single domain, also a single state variable
    (the diffusing field e.g. Temperature).
    Pass the inputs for temp evolution as a dict of diagnostics
    Eg. for temperature .. -- > give the k,rho,C_p as diagnostics field
    while pass as a dict :
        self.param['timestep'],
    self.param['timestep'],self.param['T_base'],self.param['T_top']

    Input flag use_banded_solver sets whether to use
        scipy.linalg.solve_banded
        rather than the default
        numpy.linalg.solve

        banded solver is faster but only works for 1D diffusion.
    Also note that the boundry condition is assumed to be Dirichlet type boundary condition ..

    '''
    def __init__(self,diffusion_axis=None,use_banded_solver=False,**kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.time_type = 'implicit'
        self.use_banded_solver = use_banded_solver
        if diffusion_axis is None:
            self.diffusion_axis = self._guess_diffusion_axis(self)
        else:
            self.diffusion_axis = diffusion_axis
        for dom in self.domains.values():
            delta = np.mean(dom.axes[self.diffusion_axis].delta)
            # Note that the shape of delta = 1 - points.shape, 2 - bounds.shape
        self._make_diffusion_matrix(delta)

    def _make_diffusion_matrix(self,delta):
        '''Make the array for implicit solution of the 1D heat eqn
        - Allowed variable shaped grid +
        variable thermal conductivity, density, heat capacity
        '''
        J = delta.size[0] # Size of the delta
        k_val = np.array(self.diagnostics['k']) # should be same shape as points
        rho_val = np.array(self.diagnostics['rho_c']) # should be same shape as points
        Cp_val = np.array(self.diagnostics['C_p']) # should be same shape as points

        term1a = (k_val[1:-1] + k_val[:-2])
        term1b = (k_val[1:-1] + k_val[2:])
        term3 = rho_val[1:-1]*Cp_val[1:-1]/self.param['timestep']
        term4 = delta[1:] + delta[:-1] # is same shape as k_val ..
        term5a = delta[:-1]*term4
        term5b = delta[1:]*term4
        Ka1 = (term1a/term3)/term5a
        Ka3 = (term1b/term3)/term5b
        Ka2 = Ka1 + Ka2
        add_t0 = Ka1[0]
        add_tn = Ka3[-1]
        #  Build the full banded matrix
        A = (np.diag(1. + Ka2, k=0) +
             np.diag(-Ka3[0:J-1], k=1) +
             np.diag(-Ka1[1:J], k=-1))
        self.diffTriDiag =  A
        self.add_t0 =  add_t0
        self.add_tn =  add_tn

    def _solve_implicit_banded(self,current, banded_matrix):
        #  can improve performance by storing the banded form once and not
        #  recalculating it...
        J = banded_matrix.shape[0]
        diag = np.zeros((3, J))
        diag[1, :] = np.diag(banded_matrix, k=0)
        diag[0, 1:] = np.diag(banded_matrix, k=1)
        diag[2, :-1] = np.diag(banded_matrix, k=-1)
        return solve_banded((1, 1), diag, current)

    def _implicit_solver(self):
        # Time-stepping the diffusion is just inverting this matrix problem:
        # self.T = np.linalg.solve( self.diffTriDiag, Trad )
        # Note that there should be only a single state variable - the field that is diffusing ..
        newstate = {}
        for varname, value in self.state.iteritems():
            if self.use_banded_solver:
                new_val = value[1:-1].copy()
                new_val[0] += self.param['T_base']*self.add_t0
                new_val[-1] += self.param['T_top']*self.add_tn
                newvar = self._solve_implicit_banded(new_val, self.diffTriDiag)
            else:
                new_val = value[1:-1].copy()
                new_val[0] += self.param['T_base']*self.add_t0
                new_val[-1] += self.param['T_top']*self.add_tn
                newvar = np.linalg.solve(self.diffTriDiag, new_val)
            newstate[varname][1:-1] = newvar
        return newstate

    def compute(self):
        # Time-stepping the diffusion is just inverting this matrix problem:
        # self.T = np.linalg.solve( self.diffTriDiag, Trad )
        # Note that there should be only a single state variable - the field that is diffusing ..
        newstate = self._implicit_solver()
        for varname, value in self.state.items():
            self.adjustment[varname] = newstate[varname] - value

    def _guess_diffusion_axis(self,process_or_domain):
        '''Input: a process, domain or dictionary of domains.
        If there is only one axis with length > 1 in the process or
        set of domains, return the name of that axis.
        Otherwise raise an error.'''
		axes = get_axes(process_or_domain)
		diff_ax = {}
		for axname, ax in axes.iteritems():
		    if ax.num_points > 1:
		        diff_ax.update({axname: ax})
		if len(diff_ax.keys()) == 1:
		    return diff_ax.keys()[0]
		else:
		    raise ValueError('More than one possible diffusion axis - i.e. with more than 1 num-points.')
