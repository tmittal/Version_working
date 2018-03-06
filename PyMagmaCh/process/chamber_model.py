import numpy as np
from scipy import integrate
from PyMagmaCh.process.time_dependent_process import TimeDependentProcess
from PyMagmaCh.utils import constants as const
from PyMagmaCh.utils import model_degruyter as md_deg
from PyMagmaCh.A1_domain.field import Field

## The plan is to use diagnostics for storing the calculated variables - things that
# may change during the course of the simulation .
# on the other hand, param remain constant throughout ..
# In case, any of the diagnostics are useful for later, save them as a state variable also ...

class Chamber_model(TimeDependentProcess):
    '''
    Parent class for box model of a magma chamber -
    Typically solve for P,V,T + other things - >
    This model only works for single P,T + other things for the chamber (no spatial grid in chamber ...)
    Where is the botteneck - the ODE solver used can handle only dP/dt , dT/dt etc
    Need a different setup for the case of a spatial grid in the chamber (Too complicated likely ...)
    Sequence - solve coupled ode's for the variables
    '''
    def __init__(self,chamber_shape=None,**kwargs):
        super(Chamber_model, self).__init__(**kwargs)
        self.process_type = 'explicit'
        self.chamber_shape = chamber_shape
        # this keeps track of variables to solve later, default is false for all state var
        self.solve_me ={}
        for varname, value in self.state.items():
            self.solve_me.update({varname: False})

    def func_ode(self,t,X, arg1):
        '''Specify the coupled ode functions to integrate forward ..
        This method should be over-ridden by daughter classes.'''
        pass

    def ode_solver(self):
        '''Need to specify - solve_me state variable in the daughter classes.
        Also need to make the state variables a function of time
        (i.e. start at t = 0, then move append value in the field)
        Need to be careful of the sequence in state.items()
        '''
        max_t = self.param['timestep']      # Total integration time (seconds)
        X_init = np.array([])
        X_init_var = np.array([])
        for varname, value in self.state.items():
            try:
                if (self.solve_me[varname] == True):
                    tmp1 = self.state[varname][-1] # Use the last time-step as the input
                    X_init = np.append(X_init,tmp1)
                    X_init_var = np.append(X_init_var,varname)
            except:
                pass
        tmp2 = integrate.ode(func_ode).set_integrator('dopri5',nsteps=1e8)
        tmp2.set_initial_value(X_init,0.0).set_f_params(X_init_var)
        X_new = tmp2.integrate(tmp2.t+max_t)
        state_var_update_func(X_new)

    def state_var_update_func(self,X_new):
        ''' A daughter can over-write this to save more variables if needed ...
            Note that since the sequence of things used above for X_init is the same,
            it is ok to put them back as coded below.
        '''
        counter = 0
        for varname, value in self.state.items():
            try:
                if (self.solve_me[varname] == True):
                    self.state[varname].append_val(X_new[counter])# append with each timestep
                    counter += 1
            except:
                pass

    def compute(self):
        '''Update all diagnostic quantities using current model state.
        Needs to update the tendencies - they are multipied by timestep in step_forward'''
        self.ode_solver()

# Need an input with intial states for P, T, V, eta_g,rho_m,rho_X
class Degruyter_chamber_model(Chamber_model):
    '''Parent class for box model of a magma chamber - based on Degruyter & Huber 2014.
    Solves coupled equations for P, T, V, eta_g,rho_m,rho_X
    '''
    def __init__(self,chamber_shape='spherical',depth=1e3,**kwargs):
        super(Degruyter_chamber_model, self).__init__(chamber_shape=chamber_shape,**kwargs)
        self.process_type = 'explicit'
        self.param['crustal_density'] = md_deg.crust_density
        self.plith = calc_lith_pressure(depth)
        mat_const = get_constants()
        self.param.update(mat_const) # specify the constansts for the model
        self.solve_me['P'] = True
        self.solve_me['T'] = True
        self.solve_me['V'] = True
        self.solve_me['eta_g'] = True
        self.solve_me['rho_m'] = True
        self.solve_me['rho_x'] = True
        #########################################################
        self.diagnostics['S_scale']=10. # scale factor for region to solve for heat equation
        self.diagnostics['T_S']=500. # temp at R_0*S_scale distance  - assume to be crustal temp
        # These are default functions - can be replaced by something else if needed
        self.param['func_melting_curve'] = md_deg.melting_curve_degruyter
        self.param['func_gas_density'] = md_deg.gas_density_degruyter
        self.param['func_solubility_water'] = md_deg.solubulity_curve_degruyter
        self.param['func_critical_outpar'] = md_deg.crit_outflow_degruyter
        self.param['crustal_temp_model'] = md_deg.crustal_temp_radial_degruyter
        #########################################################
        # Check that the minimal number of state variables are defined :
        assert self.state['P'] is not None
        assert self.state['T'] is not None
        assert self.state['V'] is not None
        assert self.state['eta_g'] is not None
        assert self.state['rho_m'] is not None
        assert self.state['rho_x'] is not None
        #########################################################
        ## Sets up the empty state variables to store the variables ...
        self.eta_crust = np.zeros_like(self.state['P'])
        self.R_0 = np.zeros_like(self.state['P'])
        self.meq = np.zeros_like(self.state['P'])
        self.rho_g = np.zeros_like(self.state['P'])
        self.eta_x = np.zeros_like(self.state['P'])
        self.eta_m = np.zeros_like(self.state['P'])
        self.delta_P = np.zeros_like(self.state['P'])
        self.mass_inflow = np.zeros_like(self.state['P'])
        self.mass_outflow = np.zeros_like(self.state['P'])
        #########################################################

    def calc_lith_pressure(self,depth):
        return depth*const.g_earth*self.param['crustal_density']

    def get_constants(self):
        '''
        Get material constants - can over-write this ..
        '''
        return md_deg.material_constants_degruyter()

    def func_ode(self,t,X, X_init_var):
            '''Specify the coupled ode functions to integrate forward ..
            This method should be over-ridden by daughter classes.
            '''
            P_val = np.where(X_init_var == 'P')[0][0]
            T_val = np.where(X_init_var == 'T')[0][0]
            V_val = np.where(X_init_var == 'V')[0][0]
            eta_g_val = np.where(X_init_var == 'eta_g')[0][0]
            rho_m_val = np.where(X_init_var == 'rho_m')[0][0]
            rho_x_val = np.where(X_init_var == 'rho_x')[0][0]
            dt_arry = np.zeros(6)
            func_melting_curve = self.param['func_melting_curve']
            func_gas_density = self.param['func_gas_density']
            func_solubility_water = self.param['func_solubility_water']
            crustal_temp_model = self.param['crustal_temp_model']

            eta_g = X[eta_g_val]
            rho_x = X[rho_x_val]
            rho_m = X[rho_m_val]

            eta_x,deta_x_dT,deta_x_deta_g = func_melting_curve(X[T_val],eta_g)
            rho_g, drho_g_dP,drho_g_dT = func_gas_density(X[T_val],X[P_val])
            eta_m = 1. - eta_x - eta_g
            rho_mean = eta_x*rho_x + eta_m*rho_m + eta_g*rho_g

            self.diagnostics['eta_x'] = eta_x
            self.diagnostics['eta_m'] = eta_m
            self.diagnostics['rho_g'] = rho_g
            #########################################################
            beta_mean = rho_mean/ \
                    (eta_m*rho_m/self.param['beta_m'] + \
                     eta_x*rho_x/self.param['beta_X'] + \
                     eta_g*drho_g_dP )
            alpha_mean = (eta_m*rho_m*self.param['alpha_m'] + \
                          eta_x*rho_x*self.param['alpha_X'] - \
                          eta_g*drho_g_dT )/rho_mean
            c_mean = (eta_m*rho_m*self.param['c_m'] + \
                          eta_x*rho_x*self.param['c_x'] + \
                          eta_g*rho_g*self.param['c_g'])/rho_mean
            #########################################################
            overpressure = (X[P_val] - self.plith)
            meq,dmeq_dT,dmeq_dP = func_solubility_water(X[T_val],X[P_val])
            self.diagnostics['delta_P'] = overpressure
            self.diagnostics['meq'] = meq
            #########################################################
            m_in,m_in_water,H_in = mass_in_func(t)
            m_out,m_out_water = mass_out_func(t,overpressure,eta_x,meq)
            self.diagnostics['mass_inflow'] = mass_in
            self.diagnostics['mass_outflow'] = mass_out
            self.diagnostics['mass_inflow_w'] = mass_in_water
            self.diagnostics['mass_outflow_w'] = mass_out_water

            R_0 = (X[V_val]*3./4./np.pi)**(1./3.)
            dT_dR,eta_crust =  crustal_temp_model(R_0,self.diagnostics['S_scale'],
                                                              X[T_val],T_s = self.diagnostics['T_S']),
                                                              kappa = self.param['kappa'])
            H_out_a = m_out*X[T_val]*c_mean
            H_out_b = -1.*4.*np.pi*self.param['k_crust']*R_0*R_0*dT_dR
            H_out = H_out_a + H_out_b
            self.diagnostics['eta_crust'] = eta_crust
            self.diagnostics['R_0'] = R_0

            #########################################################
            ### Matrix inversion here to get dP/dt, dT/dt, deta_g/dt
            delta_rho_xm = (rho_x- rho_m)/rho_mean
            delta_rho_gm = (rho_g - rho_m)/rho_mean
            tmp_M1 = rho_mean*X[V_val]

            a1 = (1./beta_mean + 1./self.param['beta_r'])
            b1 = (-1.*alpha_mean -self.param['alpha_r'] + deta_x_dT*delta_rho_xm)
            c1 = (delta_rho_gm + deta_x_deta_g*delta_rho_xm)
            d1 = m_in/tmp_M1 - m_out/tmp_M1 - overpressure/eta_crust
            #########################################################
            tmp_M2 = dmeq_dP/meq + 1./self.param['beta_r'] + 1./self.param['beta_m']
            tmp_M3 = meq*rho_m*eta_m/eta_g/rho_g
            tmp_M4 = dmeq_dT/meq - self.param['alpha_r'] - self.param['alpha_m'] - deta_x_dT/eta_m
            tmp_M5 = eta_g*rho_g*X[V_val]
            a2 = drho_g_dP/rho_g + 1./self.param['beta_r'] + tmp_M2*tmp_M3
            b2 = drho_g_dT/rho_g - self.param['alpha_r'] + tmp_M4*tmp_M3
            c2 = 1./eta_g - meq*rho_m*(1. + deta_x_deta_g)/eta_g/rho_g
            d2 = m_in_water/tmp_M5 - m_out_water/tmp_M5 - (1. + tmp_M3)*overpressure/eta_crust
            #########################################################
            drho_dP = eta_m*rho_m/self.param['beta_m'] + eta_x*rho_x/self.param['beta_x'] + eta_g*drho_g_dP
            drho_dT = -eta_m*rho_m*self.param['alpha_m'] - eta_x*rho_x*self.param['alpha_x'] + eta_g*drho_g_dT
                      + rho_m*( - deta_x_dT) + rho_x*deta_x_dT
            drho_etax = rho_m*( - 1./deta_x_deta_g - 1.) + rho_x + rho_g/deta_x_deta_g
            drho_etag = rho_m*( - deta_x_deta_g - 1.) + rho_x*deta_x_deta_g + rho_g
            dc_dP = (eta_m*rho_m*self.param['c_m']/self.param['beta_m'] + \
                     eta_x*rho_x*self.param['c_x']/self.param['beta_x'] + \
                     eta_g*self.param['c_g']*drho_g_dP)/rho_mean - (c_mean/rho_mean)*drho_dP
            dc_dT = (-eta_m*rho_m*self.param['c_m']*self.param['alpha_m'] - \
                     eta_x*rho_x*self.param['c_x']*self.param['alpha_x'] + \
                     eta_g*self.param['c_g']*drho_g_dT)/rho_mean - (c_mean/rho_mean)*drho_dT + \
                     deta_x_dT*(rho_x*self.param['c_x'] - rho_m*self.param['c_m'])/rho_mean - \
                     deta_x_dT*(c_mean/rho_mean)*drho_etax
            dc_deta_g = (rho_g*self.param['c_g'] - rho_m*self.param['c_m'])/rho_mean - (c_mean/rho_mean)*drho_etag + \
                        deta_x_deta_g*(rho_x*self.param['c_x'] - rho_m*self.param['c_m'])/rho_mean - \
                        deta_x_deta_g*(c_mean/rho_mean)*drho_etax
            tmp_M6 = rho_mean*c_mean*X[T_val]
            tmp_M7 = X[P_val]/tmp_M6/self.param['beta_r']
            tmp_M8 = self.param['L_m']*eta_x*rho_x/tmp_M6
            tmp_M9 = self.param['L_e']*meq*eta_m*rho_m/tmp_M6
            a3 = tmp_M7 +1./beta_mean + dc_dP/c_mean \
                - tmp_M8*(1./self.param['beta_x'] + 1./self.param['beta_r']) \
                - tmp_M9*(dmeq_dP/meq + 1./self.param['beta_m'] + 1./self.param['beta_r']) \
            b3 = -1.*self.param['alpha_r']*X[P_val]/tmp_M6 - alpha_mean + (delta_rho_xm/rho_mean)*deta_x_dT \
                + dc_dT/c_mean +1./X[T_val] - self.param['alpha_r'] \
                - tmp_M8*(-self.param['alpha_x'] - self.param['alpha_r'] + deta_x_dT/eta_x) \
                - tmp_M9*(dmeq_dT/meq - self.param['alpha_m'] - self.param['alpha_r'] - deta_x_dT/eta_m)
            c3 =  (delta_rho_gm/rho_mean) + (delta_rho_xm/rho_mean)*deta_x_deta_g \
                + dc_deta_g/c_mean \
                - self.param['L_m']*rho_x*deta_x_deta_g/tmp_M6 \
                - self.param['L_e']*meq*rho_m*(1. + deta_x_deta_g)/tmp_M6
            tmp_M10 = tmp_M6*X[V_val]
            d3 = H_in/tmp_M10 - H_out/tmp_M10 - (1. - tmp_M8 - tmp_M9)*overpressure/eta_crust
            #########################################################
            matrix1 = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]], dtype=np.float)
            matrix2 = np.array([[d1, b1, c1], [d2, b2, c2], [d3, b3, c3]], dtype=np.float)
            matrix3 = np.array([[a1, d1, c1], [a2, d2, c2], [a3, d3, c3]], dtype=np.float)
            matrix4 = np.array([[a1, b1, d1], [a2, b2, d2], [a3, b3, d3]], dtype=np.float)
            tmp_MM = np.linalg.det(matrix1)
            dt_arry[P_val] = np.linalg.det(matrix2)/tmp_MM
            dt_arry[T_val] = np.linalg.det(matrix3)/tmp_MM
            dt_arry[eta_g_val] = np.linalg.det(matrix4)/tmp_MM
            dt_arry[V_val] = X[V_val]*(overpressure/eta_crust - \
                                        self.param['alpha_r']*dt_arry[T_val] + \
                                        dt_arry[P_val]/self.param['beta_r']) \
            dt_arry[rho_m_val] = rho_m*(dt_arry[P_val]/self.param['beta_m'] - \
                                        self.param['alpha_m']*dt_arry[T_val])
            dt_arry[rho_x_val] = rho_x*(dt_arry[P_val]/self.param['beta_x'] - \
                                        self.param['alpha_x']*dt_arry[T_val])

            return dt_arry

    def mass_in_func(self,t):
        '''Specify the M_in - mass inflow rate to coupled ode functions
        This method should be over-ridden by daughter classes.
        '''
        m_in = 1. # kg/s
        m_in_water = 0.05*m_in
        eta_g_in = 0.
        T_in = 1200. # kelvin
        eta_x,deta_x_dT,deta_x_deta_g = md_deg.melting_curve_degruyter(T_in,self.plith)
        eta_m = 1. - eta_x
        rho_m0 = 2400. # kg/m^3
        rho_X0 = 2600. # kg/m^3
        rho_mean = eta_x*rho_X0 + eta_m*rho_m0
        c_mean_in = (eta_m*rho_m0*self.param['c_m'] +
                      eta_x*rho_X0*self.param['c_X'])/rho_mean
        H_in = c_mean_in*T_in*m_in
        return m_in,m_in_water,H_in

    def m_out_func(self,overpressure,eta_x,meq):
        '''Specify the M_out - mass outflow rate to coupled ode functions
        This method can be over-ridden by daughter classes.
        ### Assumptions in the paper :
        # M_in_water = 5 wt% of the melt mass inflow rate
        # M_out_water -> relative amount of water in erupted
        magma is same as chamber water wt -  i.e  M_out*M_water_chamber
        '''
        func_critical_outpar = self.param['func_critical_outpar']
        delta_Pc,eta_x_C,M_out_rate = func_critical_outpar()
        if (overpressure >= delta_Pc & eta_x <= eta_x_C):
            M_out = M_out_rate
        else :
            M_out = 0.
        m_out_water = meq*M_out
        return M_out,m_out_water

    def state_var_update_func(self,X_new):
        ''' A daughter can over-write this to save more variables if needed ...
        '''
        counter = 0
        lst_extra_var = ['eta_x','eta_m','rho_g','delta_P','mass_inflow','mass_outflow','eta_crust','R_0','meq']
        for varname, value in self.state.items():
            try:
                if (self.solve_me[varname] == True):
                    # append with each timestep
                    self.state[varname].append_val(X_new[counter])
                    counter += 1
                if varname in lst_extra_var:
                    self.state[varname].append_val(self.diagnostics[varname]) # append with each timestep
            except:
                pass


        self.param['delta_Pc'] = delta_Pc                                       # Critical Overpressure (MPa)
        self.param['eta_x_max'] = eta_x_max                                     # Locking fraction
