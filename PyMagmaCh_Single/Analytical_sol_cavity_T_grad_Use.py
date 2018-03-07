#import warnings
#import pdb
# import numpy as np
# from numpy import sqrt,pi,exp
# from scipy.special import erfc
# import input_functions as md_deg

def Analytical_sol_cavity_T_grad_Use(T_R,P_R,R_chamber,t,R_val,permeability,material) :
    #warnings.simplefilter("error", "RuntimeWarning")
    # try :
    #     tmp1 = t.shape[1]
    # except IndexError :
    #     t = np.expand_dims(t,1)
    # try :
    #     tmp1 = R_val.shape[1]
    # except IndexError :
    #     R_val = np.expand_dims(R_val,1)
    # tmp1 = md_deg.gas_density_degruyter(1000,200e6)
    # Kf = 1./(tmp1[1]/tmp1[0])
    # #% Constants in the problem are the various poro-elastic coefficients
    # if (material ==1): #%Westerly granite
    #     G = 1.5e10 #% Pa
    #     K = 2.5e10 #% Pa
    #     K_u = 3.93e10 #% Pa
    #     Ks  = 4.5e10
    #     phi = 0.01
    #     visc = 2.414*1e-5*(10.**(247.8/(600.-140))) #;% - from Rabinowicz 1998/Eldursi EPSL 2009  #% Pa s
    #     M = (Kf*Ks*Ks)/(Kf*(Ks-K) + phi*Ks*(Ks-Kf)) #% Pa
    #     S = (3.*K_u + 4*G)/(M*(3.*K + 4.*G)) #% 1/Pa
    #     c = permeability/S/visc #% m^2
    #     beta_c = 6.65e-6 # % /K (Table 11.2), W
    #     alpha_e = beta_c/S#= 3.58e5 #% N/m^2/K = beta_c/S, (Table 11.2), Westerly granite
    #     kappa_T = 1.09e-6 # % m^2/s
    #     m_d = 7.85e3/20. #%J/m^3
    #     eta_d = 2e5 #%N/m^2/K
    #     alpha_d = 6e5 #% N/m^2/K
    #     eta = 0.150 #% unitless
    #     k_T = 2.5 # W/m/K
    # if (material ==2) :#% berea Sandstone
    #     G = 6e9 #% Pa
    #     K = 8e9 #% Pa
    #     K_u =  1.4e10 #% Pa
    #     K_s = 3.6e10
    #     phi = 0.15
    #     visc = 2.414*1e-5*(10.**(247.8/(600.-140))) #;% - from Rabinowicz 1998/Eldursi EPSL 2009  #% Pa s
    #     M = (Kf*Ks*Ks)/(Kf*(Ks-K) + phi*Ks*(Ks-Kf)) #% Pa
    #     S = (3.*K_u + 4*G)/(M*(3.*K + 4*G)) #% 1/Pa
    #     c = permeability/S/visc #% m^2
    #
    #     beta_c = 4.08e-5 # % /K (Table 11.2), Westerly granite
    #     alpha_e = beta_c/S #= 2.94e5 #% N/m^2/K = beta_c/S, (Table 11.2), Westerly granite
    #     kappa_T = 1.27e-6 # % m^2/s
    #     m_d = 6.01e3/20. #%J/m^3
    #     eta_d = 1.35e4 #%N/m^2/K
    #     alpha_d = 3.6e4 #% N/m^2/K
    #     k_T = 2.24 # W/m/K
    #     eta = 0.292 #
    # S_a = m_d + alpha_d*eta_d/G
    # c_a = m_d*kappa_T/S_a
    # alpha_p = beta_c/S_a                                               # #% N/m^2/K = beta_c/S, (Table 11.2), Westerly granite
    # C_Q = np.sqrt( (c -c_a)**2. + 4.*c*c_a*alpha_p*alpha_e )           ##
    # lam1 = np.sqrt(((c +c_a) + C_Q)/2./c/c_a)                          ##
    # lam2 = np.sqrt(((c +c_a) - C_Q)/2./c/c_a)                          ##
    ##############################################
    # time_new = t[-1] - t + 1e-8                                       ## % Added the 1e-8 to make sure a min time is not zero
    # tmp_one_t = np.ones([np.shape(time_new)[0],np.shape(time_new)[1]])
    # R_steps = np.shape(R_val)[0]                                       ##
    # tmp_one_R = np.ones([np.shape(R_val)[0],np.shape(R_val)[1]])
    Diff_temp_arr = np.hstack([T_R[0],np.diff(T_R)])/k_T
    Diff_press_arr = np.hstack([P_R[0],np.diff(P_R)])*(visc/permeability)
    #term_T          = np.zeros([R_steps,np.size(T_R)])
    #term_P          = np.zeros([R_steps,np.size(T_R)])
    T_sigma_rr      = np.zeros([R_steps,np.size(T_R)])
    T_sigma_theta   = np.zeros([R_steps,np.size(T_R)])

    A_1 = Diff_temp_arr*R_chamber*R_chamber*(c-c_a+C_Q) - 2.*Diff_press_arr*alpha_p*c*R_chamber*R_chamber
    A_2 = Diff_temp_arr*R_chamber*R_chamber*(c-c_a-C_Q) - 2.*Diff_press_arr*alpha_p*c*R_chamber*R_chamber
    A_3 = (Diff_temp_arr*R_chamber*R_chamber*(c-c_a+C_Q) - 2.*Diff_press_arr*alpha_p*c*R_chamber*R_chamber)*(c-c_a-C_Q)
    A_4 = (Diff_temp_arr*R_chamber*R_chamber*(c-c_a-C_Q) - 2.*Diff_press_arr*alpha_p*c*R_chamber*R_chamber)*(c-c_a+C_Q)

    term1 = (1./C_Q)/R_val
    term1aa = lam1*(R_val - R_chamber)/2./np.sqrt(time_new.T)
    term1a = erfc(term1aa) - exp((R_val - R_chamber)/R_chamber + time_new.T/(lam1*R_chamber)**2.)*erfc(term1aa + np.sqrt(time_new.T)/(lam1*R_chamber))
    term1bb = lam2*(R_val - R_chamber)/2./np.sqrt(time_new.T)
    term1b = erfc(term1bb) - exp((R_val - R_chamber)/R_chamber + time_new.T/(lam2*R_chamber)**2.)*erfc(term1bb + np.sqrt(time_new.T)/(lam2*R_chamber))
    term_T = (-term1/2.)*(A_1*term1a - A_2*term1b)
    term_P = (term1/(4.*c*alpha_p))*(-A_3*term1a + A_4*term1b)

    #term_T_der = -(term1/2./R_val)*( A_1*erfc(term1a) - A_2*erfc(term1b) ) + (term1/2.)*(-2.*A_1*(lam1/np.sqrt(pi)/np.sqrt(time_new.T))*np.exp(-term1a**2.) + 2.*A_2*(lam2/np.sqrt(pi)/np.sqrt(time_new.T))*np.exp(-term1b**2.))

    # term_int1 = -((R_val + R_chamber)/lam1)*sqrt(time_new.T/pi)*exp(-term1a**2.) + (0.5*tmp_one_t.T*(R_val**2. - R_chamber**2.) -tmp_one_R*time_new.T/lam1**2)*erfc(term1a)
    # term_int2 = -((R_val + R_chamber)/lam2)*sqrt(time_new.T/pi)*exp(-term1b**2.) + (0.5*tmp_one_t.T*(R_val**2. - R_chamber**2.) -tmp_one_R*time_new.T/lam2**2)*erfc(term1b)
    # term_R2T = (1./C_Q/2.)*(A_1*term_int1 - A_2*term_int2)
    # term_R2P = (1./C_Q)*(1./(4.*c*alpha_p))*(A_3*term_int1 - A_4*term_int2)
    # term_int_A2a = tmp_one_R*(2.*R_chamber/lam1)*sqrt(time_new.T/pi) + tmp_one_R*time_new.T/lam1**2.#
    # term_int_A2b = tmp_one_R*(2.*R_chamber/lam2)*sqrt(time_new.T/pi) + tmp_one_R*time_new.T/lam2**2.#
    # term_A2 = (eta/C_Q/G)*(1./(4.*c*alpha_p))*(A_3*term_int_A2a - A_4*term_int_A2b) + (eta_d/C_Q/G)*(1./2.)*(A_1*term_int_A2a - A_2*term_int_A2b)  + (R_chamber**3.)*(1./4./G)*tmp_one_R*Diff_press_arr.T#
    # T_sigma_rr = -4.*eta*term_R2P/(R_val**3.) -4.*eta_d*term_R2T/(R_val**3.) -4.*G*term_A2/(R_val**3.)#
    # T_sigma_theta = 2.*eta*term_R2P/(R_val**3) +2.*eta_d*term_R2T/(R_val**3.) +2.*G*term_A2/(R_val**3) - 2.*eta*term_P - 2.*eta_d*term_T#

    # %Diff_press_arr
    # %size(term_A2),term_A2b
    # %t(end),term_P(isnan(term_P))
    T_val = np.sum(term_T,1)#
    P_val = np.sum(term_P,1)#
    # sigma_rr = np.sum(T_sigma_rr,1)#
    # sigma_theta = np.sum(T_sigma_theta,1)#
    #sigma_rr_eff = sigma_rr + P_val
    #sigma_theta_eff = sigma_theta + P_val
    #T_der = np.sum(term_T_der,1)
    #T_der[0] = T_der[1] # first value is messy and large .. so remove it ..
    P_val[P_val<1e-6] = 0
    T_val[T_val<1e-6] = 0
    return T_val,P_val#,sigma_rr,sigma_theta,T_der
