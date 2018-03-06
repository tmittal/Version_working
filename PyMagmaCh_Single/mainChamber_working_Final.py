## This is a cleaned, pythonic version of a single magma chamber model based on the Degruyter model ..

import sys
import constants as const
import numpy as np
import pdb

from assimulo.problem import Explicit_Problem
from numpy.linalg import det
import input_functions

import Analytical_sol_cavity_T_Use as Anl_sols
#from Analytical_sol_cavity_T_grad_Use import Analytical_sol_cavity_T_grad_Use

class Chamber_Problem(Explicit_Problem):
    '''
    Parent class for box model of a magma chamber -
        Typically solve for P,V,T + other things - >
        This model only works for single P,T + other things for the chamber
        (no spatial grid in chamber ...)
    Unit system - mks, --
        ==      P - In Pa
        ==      T - In Kelvin
    Where is the botteneck ?
        a. ODE solver used can handle only dP/dt , dT/dt - so an ODE solver
        b. Maybe need a different setup for the case of a spatial grid
           in the chamber (Too complicated for now since the physics is different)
    Sequence -
        solve coupled ode's for the variables
    '''

    def __init__(self,radius=1e3,depth=3e3,chamber_shape='spherical',**kwargs):
        super(Chamber_Problem, self).__init__(**kwargs)
        self.name='Spherical Magma Chamber model'
        self.param={}
        self.process_type = 'explicit'
        self.chamber_shape = chamber_shape
        self.input_functions = input_functions.Input_functions()
        self.solve_me =['P','T','eta_g','V','rho_m','rho_x']   ## List of the variables to solve in the model.
        # This a class with the default functions -
        # can be replaced by something else if needed
        self.param['heat_cond'] = 1      # Turn on/off heat conduction
        self.param['visc_relax'] = 1     # Turn on/off viscous relaxation
        self.param['press_relax'] = 1    # Turn on/off pressure diffusion
        self.param['vol_degass'] = 1     # Volatile degassing  on/off
        self.tcurrent = 0.0
        self.dt = 0.0
        self.dt_counter = 0.0
        #self.dt_crst = 3e7              # Minimum time between updating crustal solutions
        self.R_steps = 500              # radius grid points for outside the chamber
        self.param['frac_rad_Temp']   = 0.5
        self.param['frac_rad_press']  = 0.1
        self.param['frac_rad_visc']   = 0.1
        self.param['degass_frac_chm'] = 0.25
        self.param['frac_length']     = 0.2
        self.flux_in_vol = input_functions.append_me()
        self.flux_in_vol.update(1e-7)
        self.flux_out_vol = input_functions.append_me()
        self.flux_out_vol.update(1e-7)
        self.param['depth'] = depth
        self.param['T_S']=500.+400. # Background crustal temp
        self.P_list = input_functions.append_me()
        self.T_list = input_functions.append_me()
        self.P_flux_list = input_functions.append_me()
        self.T_flux_list = input_functions.append_me()
        self.times_list  = input_functions.append_me()
        self.tcurrent = 0.
        self.radius = radius
        self.permeability = 1e-20
        self.param['T_in'] = 1400.0
        self.param['eps_g_in'] = 0.0     # Gas fraction of incoming melt - gas phase ..
        self.param['m_eq_in'] = 0.05     # Volatile fraction of incoming melt
        self.param['Mdot_in'] = 1.0      # Input mass flux
        self.param['rho_m_inp'] = 2600.
        self.param['rho_x_inp'] = 3065.
        self.param['Area_conduit'] = 10.*10.
        self.param['S'] = 0.1
        self.param['outflow_model']= None # 'huppert'
        self.param['frac_cond_cool'] = 0.25 # only top of chamber cools conductively to the surface
        self.param['critical_eps_press_relax'] = 0.02 # Minimum gas fraction for pore pressure diff to be important
        self.param['delta_Pc'] = 20e6  # Critical Overpressure (MPa)
        self.param['eta_x_max'] = 0.5  # Locking fraction
        self.param['M_out_rate'] = 1e2  # kg/s
        self.param['material']        = 'Westerly_Granite' # Granite , 2 is Sandstone
        self.crust_analy_params = Anl_sols.Analytical_crust_params()
        self.T_fluid_mean = self.param['T_S']
        self.crust_analy_params.set_viscosity(self.T_fluid_mean,10*1e6) # 2nd parameter is pressure (Pa)
        self.crust_analy_params.set_constants(self.param['material'],self.permeability)
        self.tmax = -1
        self.eruption_events = {}
        self.eruption_count = 0

    def set_random_Pcrit(self):
        '''

        :return:
        '''
        delta_Pc = np.random.randint(10,50) # assume a critical overpressure btw 10 - 50 MPa
        self.param['delta_Pc'] = delta_Pc*1e6  # Critical Overpressure (MPa)
        self.input_functions.Pc = self.param['delta_Pc'] # assume a critical overpressure of 20 MPa

    def set_input_functions(self,input_fun):
        '''

        :return:
        '''
        self.input_functions= input_fun
        self.input_functions.Pc = self.param['delta_Pc'] # assume a critical overpressure of 20 MPa
        self.input_functions.eta_crit_lock = self.param['eta_x_max'] # based on cystal locking above 50 % packing ..
        self.input_functions.M_out_rate = self.param['M_out_rate'] # kg/s
        self.param['outflow_model'] = self.input_functions.outflow_model

    def set_params_crust_calcs(self,material):
        self.param['material']        = material # Granite , 2 is Sandstone
        self.crust_analy_params = Anl_sols.Analytical_crust_params()
        self.T_fluid_mean = self.param['T_S']
        self.crust_analy_params.set_viscosity(self.T_fluid_mean,self.plith) # 2nd parameter is pressure (Pa)
        self.crust_analy_params.set_constants(self.param['material'],self.permeability)

    def calc_lith_pressure(self):
        '''
        Calculate the lithospheric pressure
        :param depth:
        :return:
        '''
        return self.param['depth']*const.g_earth*self.param['crustal_density']

    def get_constants(self):
        '''
        Get material constants - can over-write this ..
        '''
        mat_const = self.input_functions.material_constants()
        self.param.update(mat_const)  # specify the constants for the model
        self.plith = self.calc_lith_pressure()

    def rhs(self,t,y,sw) :
        '''
        The right-hand-side function (rhs) for the integrator
        '''
        ######################################################################################################
        P = y[0]
        T = y[1]
        eruption = sw[4]   # This tells whether eruption is yes or no
        inside_loop = 0
        if t > self.tcurrent :
            self.dt = t-self.tcurrent
            self.dt_counter +=self.dt
            self.tcurrent = t
            inside_loop = 1
            # self.dt_counter = 0.
            self.P_list.update(P - self.plith)
            self.T_list.update(T - self.param['T_S'])
            self.times_list.update(t)
            self.T_out, self.P_out, self.sigma_rr, self.sigma_theta, self.T_der = self.crust_analy_params.Analytical_sol_cavity_T_Use(
                self.T_list.data[:self.max_count], self.P_list.data[:self.max_count], self.radius,
                self.times_list.data[:self.max_count])
            self.max_count += 1
            if np.max(self.P_out) > self.param['delta_Pc'] :
                times_evolve_p1 = np.linspace(t*1.001, t + np.pi * 1e7 * 10, 100)
                perm_chng_fac = 2.
                begin_time = self.func_evolve_overpressure(times_evolve_p1,perm_chng_fac)
            # if (self.dt_counter/self.dt_crst > 0.95) :
            #     inside_loop = 1
            #     self.dt_counter = 0.
            #     self.P_list.update(P-self.plith)
            #     self.T_list.update(T-self.param['T_S'])
            #     self.times_list.update(t)
            #     self.T_out,self.P_out,self.sigma_rr,self.sigma_theta,self.T_der = self.crust_analy_params.Analytical_sol_cavity_T_Use(self.T_list.data[:self.max_count],self.P_list.data[:self.max_count],self.radius,self.times_list.data[:self.max_count])
            #     self.max_count +=1
            #     self.T_out_p2,self.P_out_p2 = Analytical_sol_cavity_T_grad_Use(self.T_flux_list,self.P_flux_list,self.radius,self.times_list,self.R_outside,self.permeability,self.param['material'])
            #     self.T_out = self.T_out_p1 #+ self.T_out_p2
            #     self.P_out = self.P_out_p1 #+ self.P_out_p2
            #     pdb.set_trace()
            #     self.P_out_all = np.vstack([self.P_out_all,self.P_out])
            #     self.T_out_all = np.vstack([self.T_out_all,self.T_out])
            #     self.sigma_rr_all = np.vstack([self.sigma_rr_all,self.sigma_rr])
            #     self.sigma_theta_all = np.vstack([self.sigma_theta_all,self.sigma_theta])
            #     self.sigma_eff_rr_all = np.vstack([self.sigma_eff_rr_all,self.sigma_rr + self.P_out])
            #     self.sigma_eff_theta_all = np.vstack([self.sigma_eff_theta_all,self.sigma_theta+self.P_out])
            #     print(t)
            #     if  eruption ==1 :
            #         inside_loop = 1
            #         self.dt_counter = 0.
            #         self.P_list.update(P-self.plith)
            #         self.T_list.update(T-self.param['T_S'])
            #         self.times_list.update(t)
            #         self.T_out,self.P_out,self.sigma_rr,self.sigma_theta,self.T_der = Analytical_sol_cavity_T_Use(self.T_list.data[:self.max_count],self.P_list.data[:self.max_count],self.radius,self.times_list.data[:self.max_count],self.R_outside,self.permeability,self.param['material'])
            #         self.max_count +=1
            #         #self.P_out_all = np.vstack([self.P_out_all,self.P_out])
            #         #self.T_out_all = np.vstack([self.T_out_all,self.T_out])
            #         #self.sigma_rr_all = np.vstack([self.sigma_rr_all,self.sigma_rr])
            #         #self.sigma_theta_all = np.vstack([self.sigma_theta_all,self.sigma_theta])
            #         #self.sigma_eff_rr_all = np.vstack([self.sigma_eff_rr_all,self.sigma_rr + self.P_out])
            #         #self.sigma_eff_theta_all = np.vstack([self.sigma_eff_theta_all,self.sigma_theta+self.P_out])
        else :
            self.dt = 0.
        eps_g = y[2]
        V = y[3]
        dV_dP = V/self.param['beta_r']
        dV_dT = -V*self.param['alpha_r']
        rho_m = y[4]
        drho_m_dP = rho_m/self.param['beta_m']
        drho_m_dT = -rho_m*self.param['alpha_m']
        rho_x = y[5]
        drho_x_dP = rho_x/self.param['beta_x']
        drho_x_dT = -rho_x*self.param['alpha_x']
        eps_x, deps_x_dT, deps_x_deps_g = self.input_functions.melting_curve(T,P,eps_g)
        rho_g, drho_g_dP, drho_g_dT = self.input_functions.gas_density(T,P)

        rho            = (1.-eps_g-eps_x)*rho_m + eps_g*rho_g + eps_x*rho_x
        drho_dP        = (1.-eps_g-eps_x)*drho_m_dP + eps_g*drho_g_dP + eps_x*drho_x_dP
        drho_dT        = (1.-eps_g-eps_x)*drho_m_dT + eps_g*drho_g_dT + eps_x*drho_x_dT
        drho_deps_g    = -rho_m + rho_g
        drho_deps_x    = -rho_m + rho_x

        # % exsolution
        m_eq,dm_eq_dP,dm_eq_dT = self.input_functions.solubulity_curve(T,P)
        c              = ((1.-eps_g-eps_x)*rho_m*self.param['c_m'] + eps_g*rho_g*self.param['c_g'] + eps_x*rho_x*self.param['c_x'])/rho
        dc_dP          = (1./rho)*((1-eps_g-eps_x)*self.param['c_m']*drho_m_dP + eps_g*self.param['c_g']*drho_g_dP + eps_x*self.param['c_x']*drho_x_dP) - (c/rho)*drho_dP
        dc_dT          = (1./rho)*((1-eps_g-eps_x)*self.param['c_m']*drho_m_dT + eps_g*self.param['c_g']*drho_g_dT + eps_x*self.param['c_x']*drho_x_dT) - (c/rho)*drho_dT
        dc_deps_g      = (1./rho)*(-rho_m*self.param['c_m']  +  rho_g*self.param['c_g']) - (c/rho)*drho_deps_g
        dc_deps_x      = (1./rho)*(-rho_m*self.param['c_m']  +  rho_x*self.param['c_x']) - (c/rho)*drho_deps_x

        #% boundary conditions
        eps_x_in,_,_ = self.input_functions.melting_curve(self.param['T_in'],self.plith,self.param['eps_g_in'])
        rho_g_in,_,_ = self.input_functions.gas_density(self.param['T_in'],P)

        rho_m_in = self.param['rho_m_inp'] # rho_m
        rho_x_in = self.param['rho_x_inp'] # rho_x
        rho_in         = (1-self.param['eps_g_in']-eps_x_in)*rho_m_in + self.param['eps_g_in']*rho_g_in + eps_x_in*rho_x_in
        c_in           = ((1-self.param['eps_g_in']-eps_x_in)*rho_m_in*self.param['c_m'] + self.param['eps_g_in']*rho_g_in*self.param['c_g'] + eps_x_in*rho_x_in*self.param['c_x'])/rho_in #;%c;

        Mdot_in        = self.param['Mdot_in']
        Mdot_v_in      = self.param['m_eq_in']*rho_m_in*(1.-self.param['eps_g_in']-eps_x_in)*Mdot_in/rho_in + rho_g_in*self.param['eps_g_in']*Mdot_in/rho_in
        Hdot_in        = c_in*self.param['T_in']*Mdot_in

        a = (V/(4.*np.pi/3))**(1./3.)
        indx_use_P = np.where(self.R_outside <=(1.+self.param['frac_rad_press'])*a)
        indx_use_T = np.where(self.R_outside <=(1.+self.param['frac_rad_Temp'])*a)
        indx_use_visc = np.where(self.R_outside <=(1.+self.param['frac_rad_visc'])*a)
        mean_T_der_out = np.mean(self.T_der[indx_use_T])
        mean_T_out = np.mean(self.T_out[indx_use_T]) + self.param['T_S']
        mean_P_out  = np.mean(self.P_out[indx_use_P]) + self.plith
        ###############################################################
        self.T_fluid_mean = mean_T_out
        self.crust_analy_params.set_viscosity(self.T_fluid_mean,mean_P_out)
        self.crust_analy_params.set_constants(self.param['material'],self.permeability)
        ###############################################################
        mean_sigma_rr_out  = -np.mean(self.sigma_rr[indx_use_visc]) + self.plith
        visc_gas =  self.crust_analy_params.visc
        #############################################################
        if self.param['heat_cond'] == 1.:
            if t< 30e7 : # Initially the gradients are kind of large .. so may be unstable ..
                small_q =  -self.param['k_crust']*(mean_T_out-T)/(self.param['frac_rad_Temp']*a)
            else :
                small_q =  -self.param['k_crust']*mean_T_der_out
            small_q2 =  -self.param['k_crust']*(300.-self.param['T_S'])/(self.param['depth']) # conductive heat loss to the surface
            surface_area_chamber = 4.*np.pi*a**2.
            Q_out = small_q*surface_area_chamber + self.param['frac_cond_cool']*small_q2*surface_area_chamber
        elif self.param['heat_cond'] == 0.:
            Q_out = 0.
        else:
            raise NotImplementedError('heat_cond not specified')
        if np.isnan(Q_out):
            pdb.set_trace()
            raise ValueError('Q_out is NaN')
        # pdb.set_trace()
        # #############################################################
        #% set outflow conditions
        if eruption == False:
            if self.param['vol_degass'] == 1.:
                surface_area_chamber_degassing = 4.*np.pi*a**2.*self.param['degass_frac_chm']
                delta_P_grad = (P - mean_P_out)/a/self.param['frac_length']
                ################## Flux out of the chamber due to pressure gradient in the crust ..
                # Note that there is no buoyancy term since the fluid is in equilbrium (Pressure is perturbation oer bkg)
                U_og2 = (self.permeability/visc_gas)*(delta_P_grad)
                Mdot_out = eps_g*rho_g*surface_area_chamber_degassing*U_og2
                degass_hdot_water = self.param['c_g']*T*Mdot_out
                ################## Flux out of the chamber -- given the permeability of the chamber ..
                # U_og = self.input_functions.func_Uog(eps_g,eps_x,m_eq,rho_m,rho_g,T,delta_P_grad)
                # Mdot_out1 = eps_g*rho_g*surface_area_chamber_degassing*U_og
                # degass_hdot_water1 = self.param['c_g']*T*Mdot_out1
                # if np.abs(Mdot_out2) > 5 :
                #     pdb.set_trace()
                # print(Mdot_out2)
                # tmp1_sign = np.sign(Mdot_out1/Mdot_out2)
                # if (tmp1_sign == 1.0) :
                #     if (np.abs(Mdot_out2) > np.abs(Mdot_out1)) :
                #         Mdot_out =  Mdot_out2
                #         degass_hdot_water = degass_hdot_water2
                #         Q_fluid_flux_out = 0.
                #     else :
                #         Mdot_out =  Mdot_out1
                #         degass_hdot_water = degass_hdot_water1
                #         Q_fluid_flux_out = (Mdot_out - Mdot_out2)/surface_area_chamber_degassing/rho_g # extra term for the pressure equation .., m/s (i.e a velocity )
                # else :
                #         Mdot_out =  Mdot_out2 + Mdot_out1
                #         degass_hdot_water = degass_hdot_water2 +degass_hdot_water1
                #         Q_fluid_flux_out = Mdot_out1/surface_area_chamber_degassing/rho_g # extra term for the pressure equation .., m/s (i.e a velocity )
                # QH_fluid_flux_out = np.copy(degass_hdot_water)/surface_area_chamber_degassing # W/m^2
            else :
                Mdot_out = 0.
                degass_hdot_water = 0.
                # Q_fluid_flux_out = 0.
                # QH_fluid_flux_out = 0.
            Mdot_v_out = np.copy(Mdot_out) # mass loss = water loss rate
            Hdot_out = Q_out +degass_hdot_water
        elif eruption == True :
            ##########################
            surface_area_chamber_degassing = 4.*np.pi*a**2.*self.param['degass_frac_chm']
            delta_P_grad = (P - mean_P_out)/a/self.param['frac_length']
            # Note that there is no buoyancy term since the fluid is in equilbrium
            # (Pressure is perturbation over bkg)
            if self.param['vol_degass'] == 1.:
                U_og2 = (self.permeability/visc_gas)*(delta_P_grad)
                Mdot_out2 = eps_g*rho_g*surface_area_chamber_degassing*U_og2
                degass_hdot_water = self.param['c_g']*T*Mdot_out2
            else :
                Mdot_out2 = 0.
                degass_hdot_water = 0.
            # QH_fluid_flux_out = np.copy(degass_hdot_water)/surface_area_chamber_degassing # W/m^2
            ##########################
            P_buoyancy = -(rho - self.param['crustal_density'])*const.g_earth*a  # delta_rho*g*h
            Mdot_out1 = self.input_functions.crit_outflow\
                (eps_x,m_eq,T,rho,self.param['depth'],self.param['Area_conduit'],
                 self.param['S'],(P-self.plith + P_buoyancy),
                 additional_model=self.param['outflow_model'])
            # Mdot_out1 = self.param['M_out_rate'] #
            Mdot_v_out = m_eq*rho_m*(1.-eps_g-eps_x)*Mdot_out1/rho + rho_g*eps_g*Mdot_out1/rho + Mdot_out2
            Mdot_out = Mdot_out1 + Mdot_out2
            Hdot_out = c*T*Mdot_out1 + Q_out + degass_hdot_water
        else:
            raise NotImplementedError('eruption not specified')
        #############################################################
        if (inside_loop == 1) :
                #if eruption ==0 :
                # self.P_flux_list = np.hstack([self.P_flux_list,Q_fluid_flux_out])
                # self.T_flux_list = np.hstack([self.T_flux_list,QH_fluid_flux_out])
                self.flux_in_vol.update(Mdot_v_in)
                self.flux_out_vol.update(Mdot_v_out)
                #    #else :
                #         # self.P_flux_list = np.hstack([self.P_flux_list,0]) # no extra flux term ...
                #         # self.T_flux_list = np.hstack([self.T_flux_list,0]) # no extra flux term ...
                #         #pdb.set_trace()
                #         self.flux_in_vol.update(Mdot_v_in)
                #         self.flux_out_vol.update(Mdot_v_out)
        # #############################################################
        #% viscous relaxation -         #% crustal viscosity (Pa s)
        #length_met = a*(self.param['frac_rad_press']) # 1000.; %2.*a; %1000;      % Typical lengthscale for pressure diffusion ... (metamorphic aureole length-scale)
        eta_r_new = self.input_functions.crustal_viscosity(self.T_out[indx_use_visc],self.R_outside[indx_use_visc])
        if self.param['visc_relax'] == 1.:
            P_loss1 = (P-self.plith)/eta_r_new
            #P_loss1 = (P-mean_sigma_rr_out)/eta_r_new
        elif self.param['visc_relax'] == 0.:
            P_loss1 = 0.
        else:
            raise NotImplementedError('visc_relax not specified')
        # #############################################################
        # Pore Pressure relaxation -
        if self.param['press_relax'] ==1 :
            # Set that the P_loss2 is only when eps_g > 0.02
            P_loss2 = np.tanh((eps_g/self.param['critical_eps_press_relax'])*2.)*(self.permeability/visc_gas)*(P - mean_P_out)/(self.param['frac_rad_press']*a)**2.
        elif self.param['press_relax'] ==0 :
            P_loss2 = 0
        else:
            raise NotImplementedError('press_relax not specified')
        # #############################################################
        # #############################################################
        P_loss = P_loss1 + P_loss2
        # self.sigma_rr_eff = -(self.sigma_rr + self.P_out)/1e6 # in Pa
        # self.mean_sigma_rr_eff  = np.mean(self.sigma_rr_eff[indx_use_P])
        # if (eruption == 0.) and (np.abs(self.mean_sigma_rr_eff) > 20):
        #    print('EEEE')
        # #############################################################
        # #############################################################
        # #############################################################
        #  coefficients in the system of unknowns Ax = B, here x= [dP/dt dT/dt dphi/dt ...]
        #  values matrix A
        #  conservation of (total) mass
        a11 = (1/rho)*drho_dP     + (1/V)*dV_dP
        a12 = (1./rho)*drho_dT     + (1./V)*dV_dT + (1./rho)*drho_deps_x*deps_x_dT
        a13 = (1/rho)*drho_deps_g               + (1/rho)*drho_deps_x*deps_x_deps_g
        # conservation of volatile mass
        a21 = (1/rho_g)*drho_g_dP + (1/V)*dV_dP \
            + (m_eq*rho_m*(1-eps_g-eps_x))/(rho_g*eps_g)*((1/m_eq)*dm_eq_dP + (1/rho_m)*drho_m_dP + (1/V)*dV_dP)
        a22 = (1/rho_g)*drho_g_dT + (1/V)*dV_dT \
            + (m_eq*rho_m*(1-eps_g-eps_x))/(rho_g*eps_g)*((1/m_eq)*dm_eq_dT + (1/rho_m)*drho_m_dT + (1/V)*dV_dT \
            - deps_x_dT/(1-eps_g-eps_x))
        a23 = 1/eps_g - (1+deps_x_deps_g)*m_eq*rho_m/(rho_g*eps_g)
        # conservation of (total) enthalpy
        a31 = (1/rho)*drho_dP      + (1/c)*dc_dP + (1/V)*dV_dP \
            + (self.param['L_e']*rho_g*eps_g)/(rho*c*T)*((1/rho_g)*drho_g_dP + (1/V)*dV_dP) \
            - (self.param['L_m']*rho_x*eps_x)/(rho*c*T)*((1/rho_x)*drho_x_dP + (1/V)*dV_dP)
        a32 = (1/rho)*drho_dT      + (1/c)*dc_dT + (1/V)*dV_dT + 1/T \
            + (self.param['L_e']*rho_g*eps_g)/(rho*c*T)*((1/rho_g)*drho_g_dT  + (1/V)*dV_dT) \
            - (self.param['L_m']*rho_x*eps_x)/(rho*c*T)*((1/rho_x)*drho_x_dT + (1/V)*dV_dT)  \
            + ((1/rho)*drho_deps_x + (1/c)*dc_deps_x - (self.param['L_m']*rho_x)/(rho*c*T))*deps_x_dT
        a33 = (1/rho)*drho_deps_g  + (1/c)*dc_deps_g \
            + (self.param['L_e']*rho_g)/(rho*c*T) \
            + ((1/rho)*drho_deps_x + (1/c)*dc_deps_x - (self.param['L_m']*rho_x)/(rho*c*T))*deps_x_deps_g
        # values vector B
        # conservation of (total) mass
        b1  =  (Mdot_in - Mdot_out)/(rho*V) - P_loss
        # conservation of volatile mass
        b2  =  (Mdot_v_in - Mdot_v_out)/(rho_g*eps_g*V) - P_loss*(1+(m_eq*rho_m*(1-eps_g-eps_x))/(rho_g*eps_g))
        # conservation of (total) enthalpy
        b3  =  (Hdot_in - Hdot_out)/(rho*c*T*V) - P_loss*(1-(self.param['L_m']*rho_x*eps_x)/(rho*c*T)+(self.param['L_e']*rho_g*eps_g)/(rho*c*T) - P/(rho*c*T));
        # set up matrices to solve using Cramer's rule
        A          = np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        A_P        = np.array([[b1,a12,a13],[b2,a22,a23],[b3,a32,a33]])
        A_T        = np.array([[a11,b1,a13],[a21,b2,a23],[a31,b3,a33]])
        A_eps_g    = np.array([[a11,a12,b1],[a21,a22,b2],[a31,a32,b3]])
        det_A = det(A)
        dP_dt = det(A_P)/det_A
        dT_dt = det(A_T)/det_A
        deps_g_dt = det(A_eps_g)/det_A
        dV_dt          = dV_dP*dP_dt + dV_dT*dT_dt + V*P_loss
        drho_m_dt      = drho_m_dP*dP_dt + drho_m_dT*dT_dt
        drho_x_dt      = drho_x_dP*dP_dt + drho_x_dT*dT_dt
        dydz = np.zeros(6)
        # column vector
        dydz[0] = dP_dt
        dydz[1] = dT_dt
        dydz[2] = deps_g_dt
        dydz[3] = dV_dt
        dydz[4] = drho_m_dt
        dydz[5] = drho_x_dt
        return dydz

    def state_events(self,t,y,sw):
        '''
        Local Variables:
        direction, value2, P_crit, isterminal, eruption, Q_out, value, P, value4, value1c, T, value1a,
         y, value3, eps_g, P_0, value1b
        '''
        P = y[0]
        T = y[1]
        eps_g = y[2]
        V = y[3]
        rho_m = y[4]
        rho_x = y[5]
        eps_x,_,_  = self.input_functions.melting_curve(T,P,eps_g)
        rho_g,_,_  = self.input_functions.gas_density(T,P)
        rho = (1.-eps_g-eps_x)*rho_m + eps_g*rho_g + eps_x*rho_x
        value1a = eps_g  #% Detect eps_g approaching 0
        value1b = eps_x
        value1c = eps_x/(1.-eps_g)-0.8 # 80% crystals in magma crystal mixture ..
        value2 = eps_x-self.param['eta_x_max']
        a = (V/(4.*np.pi/3))**(1./3.)
        P_buoyancy = -(rho - self.param['crustal_density'])*const.g_earth*a  # delta_rho*g*h
        if sw[4] : # is True (eruption)
              value3 = self.plith - P
        else : # no eruption
              value3 = (P-self.plith + P_buoyancy) - self.param['delta_Pc']
        value = np.array([value1a, value1b, value1c,value2,value3])
        #isterminal = np.array([1, 1, 1, 1, 1,1]) #% Stop the integration
        #direction = np.array([0, 0, 0, 1, 1, 0])
        return value


    #Helper function for handle_event
    def event_switch(self,solver, event_info):
        """
        Turns the switches.
        """
        for i in range(len(event_info)): #Loop across all event functions
            if event_info[i] != 0:
                solver.sw[i] = not solver.sw[i] #Turn the switch

    def handle_event(self,solver, event_info):
        """
        Event handling. This functions is called when Assimulo finds an event as
        specified by the event functions.
        """
        event_info = event_info[0]   #We only look at the state events information.
        while True:  #Event Iteration
            self.event_switch(solver, event_info) #Turns the switches
            b_mode = self.state_events(solver.t, solver.y, solver.sw)
            self.init_mode(solver) #Pass in the solver to the problem specified init_mode
            a_mode = self.state_events(solver.t, solver.y, solver.sw)
            event_info = self.check_eIter(b_mode, a_mode)
            #print(event_info)
            if not True in event_info: #sys.exit()s the iteration loop
                break

    def init_mode(self,solver):
        """
        Initialize with the new conditions.
        """
        ## No change in the initial conditions (i.e. the values of the parameters when the eruption initiates .. - like P,V, ... T)
        ## Maybe can use it to switch pore-pressure degassing on/off during eruption
        #solver.y[1] = (-1.0 if solver.sw[1] else 3.0)
        #solver.y[2] = (0.0 if solver.sw[2] else 2.0)
        ## Gas (eps_g = zero), eps_x is zero, too many crystals, 50 % crystallinity,eruption (yes/no)
        if (solver.sw[3] ==True) and (solver.sw[4] == True):
            print('critical pressure - {:f} MPa reached but eps_x>{:.0f}'.format(self.param['delta_Pc']/1e6,self.param['eta_x_max']))
            sys.exit(solver.t_sol)
        if solver.sw[0]:
            print('eps_g became 0.')
        if True in solver.sw[0:4] :
            print('Reached the end of the calculations since : ')
            if solver.sw[1] :
                print('eps_x became 0.')
            elif solver.sw[2] :
                print('eps_x/(1-eps_g) became 0.8')
            elif solver.sw[3] :
                print('eps_x became {:f}'.format(self.param['eta_x_max']))
            sys.exit(solver.t_sol)
        if (solver.sw[4] == True) :
            self.eruption_events[str(self.eruption_count)] = solver.t/np.pi/1e7,solver.y
            self.eruption_count += 1
        return 0

    #Helper function for handle_event
    def check_eIter(self,before, after):
        """
        Helper function for handle_event to determine if we have event
        iteration.
         Input: Values of the event indicator functions (state_events)
         before and after we have changed mode of operations.
        """
        eIter = [False]*len(before)
        for i in range(len(before)):
             if (before[i] < 0.0 and after[i] > 0.0) or (before[i] > 0.0 and after[i] < 0.0):
                 eIter[i] = True
        return eIter

    def func_evolve_init_cond(self,T_0):
        '''
        Calculate the initial evolution of the system - regularize the pore pressure condition
        :param self:
        :return:
        '''
        ### First evolve the solution to a 1 yr (a few points is ok since everything is analytical ..)
        times_evolve_p1 = np.linspace(1e3,np.pi*1e7,10)
        for i in times_evolve_p1:
            self.P_list.update(0.)
            self.T_list.update(T_0 - self.param['T_S'])
            self.times_list.update(i)
            self.T_out, self.P_out, self.sigma_rr, self.sigma_theta, self.T_der = \
                self.crust_analy_params.Analytical_sol_cavity_T_Use(self.T_list.data[:self.max_count],
                                                                       self.P_list.data[:self.max_count],
                                                                       self.radius, self.times_list.data[:self.max_count])
            #print(i,np.max(self.P_out) / self.param['delta_Pc'])
            self.max_count += 1
        #print(i,np.max(self.P_out) / self.param['delta_Pc'])
        if np.max(self.P_out) > 0.8*self.param['delta_Pc'] :
            times_evolve_p1 = np.linspace(1.5 * np.pi * 1e7, np.pi * 1e7 * 1e2, 100)
            perm_chng_fac = 1.25
            begin_time = self.func_evolve_overpressure(times_evolve_p1,perm_chng_fac)
            return begin_time*1.01
        else :
            begin_time = np.pi*1e7
            self.extra_vals = 10
            return begin_time*1.01

    def func_evolve_overpressure(self,times_evolve_p1,perm_chng_fac):
        '''
        Regularize the pore pressure condition
        :param self:
        :return:
        '''
        ### First evolve the solution to a 1 yr (a few points is ok since everything is analytical ..)
        perm_init = self.permeability
        excess_press = True
        i_count = 0
        self.perm_evl_init = np.append(self.perm_evl_init,perm_init)
        self.perm_evl_init_time = np.append(self.perm_evl_init,perm_init)
        P_cond = self.P_list.data[self.max_count-1] ## Keep this constant with time for the subsequent evolution ..
        T_cond = self.T_list.data[self.max_count-1] ## Keep this constant with time for the subsequent evolution ..
        while excess_press :
            self.P_list.update(P_cond)
            self.T_list.update(T_cond)
            self.times_list.update(times_evolve_p1[i_count])
            self.T_out, self.P_out, self.sigma_rr, self.sigma_theta, self.T_der = self.crust_analy_params.Analytical_sol_cavity_T_Use\
                (self.T_list.data[:self.max_count],self.P_list.data[:self.max_count],self.radius, self.times_list.data[:self.max_count])
            self.max_count += 1
            i_count += 1
            #plt.plot(self.R_outside,self.P_out/1e6)
            #plt.pause(.2)
            if np.max(self.P_out) < 0.99*self.param['delta_Pc'] :
                excess_press = False
            self.permeability = self.permeability*perm_chng_fac
            self.set_params_crust_calcs('Westerly_Granite')
            self.crust_analy_params.set_misc_grids(self.R_outside)
            self.perm_evl_init = np.append(self.perm_evl_init,self.permeability)
        #pdb.set_trace()
        self.perm_evl_init_time = np.append(self.perm_evl_init_time,times_evolve_p1[0:i_count])
        self.permeability = perm_init
        begin_time = self.perm_evl_init_time[-1]
        self.extra_vals = 10 + i_count
        print('func_evolve_overpressure')
        return begin_time*1.01
