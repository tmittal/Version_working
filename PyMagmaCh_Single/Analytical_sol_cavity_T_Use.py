#import warnings
import pdb
import numpy as np
#from numpy import np.sqrt,pi,exp
from scipy.special import erfc
from iapws import IAPWS95

class Analytical_crust_params(object):
    def __init__(self):
        self.name = 'parameters for the Analytical_solution'
        self.T_fluid = 750.
        self.P_fluid = 10.*1e6
        self.set_viscosity(self.T_fluid,10*1e6)
        self.set_constants('Westerly_Granite',1e-19)
        self.fluid_prop = IAPWS95(P=self.P_fluid/1e6,T=self.T_fluid) # T_fluid in Kelvin, P in MPa

    def set_viscosity(self,T_fluid,P_fluid):
        self.T_fluid = T_fluid
        self.P_fluid = P_fluid
        #self.visc = 2.414 * 1e-5 * (10. ** (247.8 / (self.T_fluid - 140)))  # ;% - from Rabinowicz 1998/Eldursi EPSL 2009  #% Pa
        self.fluid_prop = IAPWS95(P=self.P_fluid/1e6, T=self.T_fluid)  # T_fluid in Kelvin, P in MPa
        self.visc = self.fluid_prop.mu # Dynamic viscosity [Pa s]

    def set_constants(self,material,permeability):
        self.material = material
        self.Kf = self.fluid_prop.Ks*1e6 # Adiabatic bulk modulus (output was in MPa)
        self.permeability = permeability
        if (material =='Westerly_Granite'): #%
            self.G = 1.5e10 #% Pa
            self.K = 2.5e10 #% Pa
            self.K_u = 3.93e10 #% Pa
            self.Ks  = 4.5e10
            self.phi = 0.01
            self.M = (self.Kf*self.Ks*self.Ks)/(self.Kf*(self.Ks-self.K) + self.phi*self.Ks*(self.Ks-self.Kf)) #% Pa
            self.S = (3.*self.K_u + 4*self.G)/(self.M*(3.*self.K + 4.*self.G)) #% 1/Pa
            self.c = self.permeability/self.S/self.visc #% m^2
            self.beta_c = 6.65e-6 # % /K (Table 11.2), W
            self.alpha_e = self.beta_c/self.S#= 3.58e5 #% N/m^2/K = beta_c/S, (Table 11.2), Westerly granite
            self.kappa_T = 1.09e-6 # % m^2/s
            self.m_d = 7.85e3/20. #%J/m^3
            self.eta_d = 2e5 #%N/m^2/K
            self.alpha_d = 6e5 #% N/m^2/K
            self.eta = 0.150 #% unitless
            self.k_T = 2.5 # W/m/K
        elif (material =='Berea_Sandstone') :#% berea Sandstone
            self.G = 6e9 #% Pa
            self.K = 8e9 #% Pa
            self.K_u =  1.4e10 #% Pa
            self.Ks = 3.6e10
            self.phi = 0.15
            self.M = (self.Kf*self.Ks*self.Ks)/(self.Kf*(self.Ks-self.K) + self.phi*self.Ks*(self.Ks-self.Kf)) #% Pa
            self.S = (3.*self.K_u + 4*self.G)/(self.M*(3.*self.K + 4*self.G)) #% 1/Pa
            self.c = self.permeability/self.S/self.visc #% m^2
            self.beta_c = 4.08e-5 # % /K (Table 11.2), Westerly granite
            self.alpha_e = self.beta_c/self.S #= 2.94e5 #% N/m^2/K = beta_c/S, (Table 11.2), Westerly granite
            self.kappa_T = 1.27e-6 # % m^2/s
            self.m_d = 6.01e3/20. #%J/m^3
            self.eta_d = 1.35e4 #%N/m^2/K
            self.alpha_d = 3.6e4 #% N/m^2/K
            self.k_T = 2.24 # W/m/K
            self.eta = 0.292 #
        else:
            raise NotImplementedError('material not specified')
        self.S_a = self.m_d + self.alpha_d*self.eta_d/self.G
        self.c_a = self.m_d*self.kappa_T/self.S_a
        self.alpha_p = self.beta_c/self.S_a                                               # #% N/m^2/K = beta_c/S, (Table 11.2), Westerly granite
        self.C_Q = np.sqrt( (self.c -self.c_a)**2. + 4.*self.c*self.c_a*self.alpha_p*self.alpha_e )           ##
        self.lam1 = np.sqrt(((self.c +self.c_a) + self.C_Q)/2./self.c/self.c_a)                          ##
        self.lam2 = np.sqrt(((self.c +self.c_a) - self.C_Q)/2./self.c/self.c_a)                          ##

    def set_misc_grids(self,R_val):
        try :
            tmp1 = R_val.shape[1]
            del tmp1
        except IndexError :
            R_val = np.expand_dims(R_val,1)
        self.R_val = R_val
        self.R_steps = np.shape(R_val)[0]                                       ##
        self.tmp_one_R = np.ones([np.shape(R_val)[0],np.shape(R_val)[1]])
        self.term1 = (1./self.C_Q)/self.R_val

    def Analytical_sol_cavity_T_Use(self,T_R,P_R,R_chamber,t,T_flux,P_flux) :
        try :
            tmp1 = t.shape[1]
            del tmp1
        except IndexError :
            t = np.expand_dims(t,1)
        ##############################################
        time_new = t[-1] - t + 1.0                                      ## % Added the 1 to make sure a min time is not zero
        tmp_one_t = np.ones([np.shape(time_new)[0],np.shape(time_new)[1]])
        #Diff_temp_arr = np.hstack([T_R[0],np.diff(T_R)])
        #Diff_press_arr = np.hstack([P_R[0],np.diff(P_R)])
        Diff_temp_arr = T_R.copy()
        Diff_temp_arr[1:] = T_R[1:] - T_R[:-1]
        Diff_press_arr = P_R.copy()
        Diff_press_arr[1:] = P_R[1:] - P_R[:-1]
        ############################################
        Diff_grad_temp_arr = T_flux.copy()/self.k_T
        Diff_grad_temp_arr[1:] = T_flux[1:] - T_flux[:-1]
        Diff_grad_press_arr = P_flux.copy()*(self.visc/self.permeability)
        Diff_grad_press_arr[1:] = P_flux[1:] - P_flux[:-1]

        time_new = time_new.T
        tmp_one_t = tmp_one_t.T
        sqrt_time_new = np.sqrt(time_new)

        #term_T          = np.zeros([self.R_steps,np.size(T_R)])
        #term_P          = np.zeros([self.R_steps,np.size(T_R)])
        #T_sigma_rr      = np.zeros([self.R_steps,np.size(T_R)])
        #T_sigma_theta   = np.zeros([self.R_steps,np.size(T_R)])
        term1a = self.lam1*(self.R_val - R_chamber)/2./sqrt_time_new
        term1b = self.lam2*(self.R_val - R_chamber)/2./sqrt_time_new
        Erfc_t1a = erfc(term1a)
        Erfc_t1b = erfc(term1b)
        exp_term1a = np.exp(-term1a**2.)
        exp_term1b = np.exp(-term1b**2.)
        msc_fact1 = (self.lam1/np.sqrt(np.pi)/sqrt_time_new)
        msc_fact2 = (self.lam2/np.sqrt(np.pi)/sqrt_time_new)
        sqrt_time_new_term = sqrt_time_new/np.sqrt(np.pi)

        #################################################################
        term1_difft_R_chm_grad = Diff_grad_temp_arr*R_chamber*R_chamber
        term1_diffP_R_chm_grad = Diff_grad_press_arr*R_chamber*R_chamber

        A_1_grad = term1_difft_R_chm_grad*(self.c - self.c_a + self.C_Q) - 2.*term1_diffP_R_chm_grad*self.alpha_p*self.c
        A_2_grad = term1_difft_R_chm_grad*(self.c - self.c_a - self.C_Q) - 2.*term1_diffP_R_chm_grad*self.alpha_p*self.c
        A_3_grad = (A_1_grad) * (self.c - self.c_a - self.C_Q)
        A_4_grad = (A_2_grad) * (self.c - self.c_a + self.C_Q)

        term1a_grad = Erfc_t1a - np.exp((self.R_val - R_chamber)/R_chamber + time_new/(self.lam1*R_chamber)** 2.)*erfc(term1a + sqrt_time_new/(self.lam1*R_chamber))
        term1b_grad = Erfc_t1b - np.exp((self.R_val - R_chamber)/R_chamber + time_new/(self.lam2*R_chamber)** 2.)*erfc(term1b + sqrt_time_new/(self.lam2*R_chamber))
        term_T_grad = (-self.term1*0.5)*(A_1_grad*term1a_grad - A_2_grad*term1b_grad)
        term_P_grad = (self.term1/(4.*self.c*self.alpha_p))*(-A_3_grad*term1a_grad + A_4_grad*term1b_grad)

        # a_1 = self.lam1*(self.R_val - R_chamber)
        # b_1 = 1./self.lam1/R_chamber
        # term_int1_grad = ((2. - b_1*self.lam1*tmp_one_t*(self.R_val + R_chamber))/(b_1*self.lam1**2.))*sqrt_time_new_term*np.exp(-a_1/4./time_new) +\
        #                  ((1. - b_1*self.lam1*tmp_one_t*self.R_val)/((b_1*self.lam1)**2.))*np.exp(b_1*(tmp_one_t*a_1 + b_1*self.tmp_one_R*time_new))*erfc((tmp_one_t*a_1+2.*b_1*self.tmp_one_R*time_new)/2./sqrt_time_new) - \
        #                  ((2.*(1.- b_1*self.lam1*tmp_one_t*self.R_val + self.tmp_one_R*time_new*b_1**2.) - (b_1**2.*self.lam1**2.)*tmp_one_t*(self.R_val**2. - R_chamber**2.))/2./(b_1*self.lam1)**2. )*erfc(tmp_one_t*a_1/2./sqrt_time_new)
        # a_1 = 0.
        # term_int1_grad_0 = ((2. - b_1*self.lam1*tmp_one_t*(R_chamber + R_chamber))/(b_1*self.lam1**2.))*sqrt_time_new_term*np.exp(-a_1/4./time_new) +\
        #                  ((1. - b_1*self.lam1*tmp_one_t*R_chamber)/((b_1*self.lam1)**2.))*np.exp(b_1*(tmp_one_t*a_1 + b_1*self.tmp_one_R*time_new))*erfc((tmp_one_t*a_1+2.*b_1*self.tmp_one_R*time_new)/2./sqrt_time_new) - \
        #                  ((2.*(1.- b_1*self.lam1*tmp_one_t*R_chamber + self.tmp_one_R*time_new*b_1**2.) - (b_1**2.*self.lam1**2.)*tmp_one_t*(R_chamber**2. - R_chamber**2.))/2./(b_1*self.lam1)**2. )*erfc(tmp_one_t*a_1/2./sqrt_time_new)
        #
        # a_1 = self.lam2*(self.R_val - R_chamber)
        # b_1 = 1./self.lam2/R_chamber
        # term_int2_grad = ((2. - b_1*self.lam2*tmp_one_t*(self.R_val + R_chamber))/(b_1*self.lam2**2.))*sqrt_time_new_term*np.exp(-a_1/4./time_new) +\
        #                  ((1. - b_1*self.lam2*tmp_one_t*self.R_val)/((b_1*self.lam2)**2.))*np.exp(b_1*(tmp_one_t*a_1 + b_1*self.tmp_one_R*time_new))*erfc((tmp_one_t*a_1+2.*b_1*self.tmp_one_R*time_new)/2./sqrt_time_new) - \
        #                  ((2.*(1.- b_1*self.lam2*tmp_one_t*self.R_val + self.tmp_one_R*time_new*b_1**2.) - (b_1**2.*self.lam2**2.)*tmp_one_t*(self.R_val**2. - R_chamber**2.))/2./(b_1*self.lam2)**2. )*erfc(tmp_one_t*a_1/2./sqrt_time_new)
        # a_1 = 0.
        # term_int2_grad_0 = ((2. - b_1*self.lam2*tmp_one_t*(R_chamber + R_chamber))/(b_1*self.lam2**2.))*sqrt_time_new_term*np.exp(-a_1/4./time_new) +\
        #                  ((1. - b_1*self.lam2*tmp_one_t*R_chamber)/((b_1*self.lam2)**2.))*np.exp(b_1*(tmp_one_t*a_1 + b_1*self.tmp_one_R*time_new))*erfc((tmp_one_t*a_1+2.*b_1*self.tmp_one_R*time_new)/2./sqrt_time_new) - \
        #                  ((2.*(1.- b_1*self.lam2*tmp_one_t*R_chamber + self.tmp_one_R*time_new*b_1**2.) - (b_1**2.*self.lam2**2.)*tmp_one_t*(R_chamber**2. - R_chamber**2.))/2./(b_1*self.lam2)**2. )*erfc(tmp_one_t*a_1/2./sqrt_time_new)
        # term_R2T_grad = (-1./self.C_Q/2.)*(A_1_grad*term_int1_grad - A_2_grad*term_int2_grad)
        # term_R2P_grad = (1./self.C_Q)*(1./(4.*self.c*self.alpha_p))*(A_3_grad*term_int1_grad - A_4_grad*term_int2_grad)
        # term_R2T_grad_0 = (-1./self.C_Q/2.)*(A_1_grad*term_int1_grad_0 - A_2_grad*term_int2_grad_0)
        # term_R2P_grad_0 = (1./self.C_Q)*(1./(4.*self.c*self.alpha_p))*(A_3_grad*term_int1_grad_0 - A_4_grad*term_int2_grad_0)
        # term_A2_grad = -(self.eta/self.G)*term_R2P_grad_0 - (self.eta_d/self.G)*term_R2T_grad_0
        # T_sigma_rr_grad = -4.*self.eta*term_R2P_grad/(self.R_val**3.) -4.*self.eta_d*term_R2T_grad/(self.R_val**3.) -4.*self.G*term_A2_grad/(self.R_val**3.)#
        # T_sigma_theta_grad = 2.*self.eta*term_R2P_grad/(self.R_val**3) +2.*self.eta_d*term_R2T_grad/(self.R_val**3.) +2.*self.G*term_A2_grad/(self.R_val**3) - 2.*self.eta*term_P_grad - 2.*self.eta_d*term_T_grad#
        # # del term_R2T_grad,term_R2P_grad,term_R2T_grad_0,term_R2P_grad_0,term_A2_grad,term_int1_grad,term_int2_grad,term_int1_grad_0,term_int2_grad_0

        #################################################################
        #################################################################

        term1_difft_R_chm = Diff_temp_arr*R_chamber
        term1_diffP_R_chm = Diff_press_arr*R_chamber
        A_1 =  term1_difft_R_chm*(self.c-self.c_a+self.C_Q) - 2.*term1_diffP_R_chm*self.alpha_p*self.c
        A_2 =  term1_difft_R_chm*(self.c-self.c_a-self.C_Q) - 2.*term1_diffP_R_chm*self.alpha_p*self.c
        A_3 = (A_1)*(self.c-self.c_a-self.C_Q)
        A_4 = (A_2)*(self.c-self.c_a+self.C_Q)

        term_T = (self.term1/2.)*( A_1*Erfc_t1a - A_2*Erfc_t1b )
        term_P = (self.term1/(4.*self.c*self.alpha_p))*( A_3*Erfc_t1a - A_4*Erfc_t1b )

        term_T_der = -(self.term1/2./self.R_val)*(A_1*Erfc_t1a - A_2*Erfc_t1b) + \
                     (self.term1/2.)*(-A_1*msc_fact1*exp_term1a + A_2*msc_fact2*exp_term1b)
        # term_P_der = -(self.term1/(4.*self.c*self.alpha_p)/self.R_val)*(A_3*Erfc_t1a - A_4*Erfc_t1b) + \
        #              (self.term1/(4.*self.c*self.alpha_p))*(-A_3*msc_fact1*exp_term1a + A_4*msc_fact1*exp_term1a)

        term_int1 = -((self.R_val + R_chamber)/self.lam1)*sqrt_time_new_term*exp_term1a + \
                    (0.5*tmp_one_t*(self.R_val**2. - R_chamber**2.) -self.tmp_one_R*time_new/self.lam1**2)*Erfc_t1a
        term_int2 = -((self.R_val + R_chamber)/self.lam2)*sqrt_time_new_term*exp_term1b + \
                    (0.5*tmp_one_t*(self.R_val**2. - R_chamber**2.) -self.tmp_one_R*time_new/self.lam2**2)*Erfc_t1b
        term_R2T = (1./self.C_Q/2.)*(A_1*term_int1 - A_2*term_int2)
        term_R2P = (1./self.C_Q)*(1./(4.*self.c*self.alpha_p))*(A_3*term_int1 - A_4*term_int2)
        term_int_A2a = self.tmp_one_R*(2.*R_chamber/self.lam1)*sqrt_time_new_term + self.tmp_one_R*time_new/self.lam1**2.#
        term_int_A2b = self.tmp_one_R*(2.*R_chamber/self.lam2)*sqrt_time_new_term + self.tmp_one_R*time_new/self.lam2**2.#
        term_A2 = (self.eta/self.C_Q/self.G)*(1./(4.*self.c*self.alpha_p))*(A_3*term_int_A2a - A_4*term_int_A2b) + (self.eta_d/self.C_Q/self.G)*(1./2.)*(A_1*term_int_A2a - A_2*term_int_A2b)  + (R_chamber**3.)*(1./4./self.G)*self.tmp_one_R*Diff_press_arr.T#
        T_sigma_rr = -4.*self.eta*term_R2P/(self.R_val**3.) -4.*self.eta_d*term_R2T/(self.R_val**3.) -4.*self.G*term_A2/(self.R_val**3.)#
        T_sigma_theta = 2.*self.eta*term_R2P/(self.R_val**3) +2.*self.eta_d*term_R2T/(self.R_val**3.) +2.*self.G*term_A2/(self.R_val**3) - 2.*self.eta*term_P - 2.*self.eta_d*term_T#

        T_val = np.sum(term_T,1) + np.sum(term_T_grad,1) #
        P_val = np.sum(term_P,1) + np.sum(term_P_grad,1) #
        sigma_rr = np.sum(T_sigma_rr,1)#
        sigma_theta = np.sum(T_sigma_theta,1)#
        # sigma_rr = T_val*0.0
        # sigma_theta = T_val*0.0
        #sigma_rr_eff = sigma_rr + P_val
        #sigma_theta_eff = sigma_theta + P_val
        T_der = np.sum(term_T_der,1)
        T_der[0] = T_der[1] # first value is messy and large .. so remove it ..
        #sigma_rr_grad = np.sum(T_sigma_rr_grad,1)#
        # if np.max(sigma_rr_grad) > 8000. : -- Currently in the numerical noise
        #     print(np.max(sigma_rr_grad), np.max(sigma_rr))
        #     pdb.set_trace()
        return T_val,P_val,sigma_rr,sigma_theta,T_der