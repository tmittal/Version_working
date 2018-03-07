### This is the testing piece for the initial temperature shock piece ..

from assimulo.solvers import CVode,LSODAR
import sys
import numpy as np
import pylab as plt
from plot_mainChamber import plot_mainChamber
from mainChamber_working_Final import Chamber_Problem
import input_functions as inp

T_S = float(sys.argv[1]) #'1p17_diff_'
perm_val = float(sys.argv[2])
#print(sys.argv)

#% set the mass inflow rate
mdot          = 10.   #;          % mass inflow rate (kg/s) #% use global variable
depth  =  8000.
with_plots=True

##############################################
#% time
end_time       = 3e7*1e5#; % maximum simulation time in seconds
begin_time     = 0 #; % initialize time
##############################################
T_0 = 1200  # ;       % initial chamber temperature (K)

def func_set_system():
    ##############################################
    #% initial conditions
    P_0           = depth*9.8*2600.           #;      % initial chamber pressure (Pa)
    #T_0           = 1200            #;       % initial chamber temperature (K)
    eps_g0        = 0.04            #;       % initial gas volume fraction
    rho_m0        = 2600            #;       % initial melt density (kg/m^3)
    rho_x0        = 3065            #;       % initial crystal density (kg/m^3)
    a             = 3000            #;       % initial radius of the chamber (m)
    V_0           = (4.*np.pi/3.)*a**3.  #; % initial volume of the chamber (m^3)

    ##############################################
    ##############################################
    IC = np.array([P_0, T_0, eps_g0, V_0, rho_m0, rho_x0])  # % store initial conditions
    ## Gas (eps_g = zero), eps_x is zero, too many crystals, 50 % crystallinity,eruption (yes/no)
    sw0 = [False,False,False,False,False]

    ##############################################
    #% error tolerances used in ode method
    dt = 30e7
    N  = int(round((end_time-begin_time)/dt))
    ##############################################

    #Define an Assimulo problem
    exp_mod = Chamber_Problem(depth=depth,t0=begin_time,y0=IC,sw0=sw0)
    exp_mod.param['T_in'] = 1200.
    exp_mod.param['eps_g_in'] = 0.0    # Gas fraction of incoming melt - gas phase ..
    exp_mod.param['m_eq_in'] = 0.03    # Volatile fraction of incoming melt
    exp_mod.param['Mdot_in']    = mdot
    exp_mod.param['eta_x_max'] = 0.64                                     # Locking fraction
    exp_mod.param['delta_Pc']   = 20e6
    exp_mod.tcurrent = begin_time
    exp_mod.radius = a
    exp_mod.permeability = perm_val
    exp_mod.R_steps = 5500
    exp_mod.dt_init = dt
    inp_func1 = inp.Input_functions_Degruyer()
    exp_mod.set_input_functions(inp_func1)
    exp_mod.get_constants()
    exp_mod.param['T_S'] = T_S
    #################
    exp_mod.R_outside = np.linspace(a,2.*a,exp_mod.R_steps)
    exp_mod.set_params_crust_calcs('Westerly_Granite')
    exp_mod.crust_analy_params.set_misc_grids(exp_mod.R_outside)
    exp_mod.T_out_all =np.array([exp_mod.R_outside*0.])
    exp_mod.P_out_all =np.array([exp_mod.R_outside*0.])
    exp_mod.sigma_rr_all    = np.array([exp_mod.R_outside*0.])
    exp_mod.sigma_theta_all = np.array([exp_mod.R_outside*0.])
    exp_mod.sigma_eff_rr_all = np.array([exp_mod.R_outside*0.])
    exp_mod.sigma_eff_theta_all = np.array([exp_mod.R_outside*0.])
    exp_mod.max_count = 1 # counting for the append me arrays ..

    P_0 = exp_mod.plith
    exp_mod.P_list.update(0.)
    exp_mod.T_list.update(T_0-exp_mod.param['T_S'])
    exp_mod.P_flux_list.update(0)
    exp_mod.T_flux_list.update(0)
    exp_mod.times_list.update(1e-7)
    exp_mod.T_out,exp_mod.P_out,exp_mod.sigma_rr,exp_mod.sigma_theta,exp_mod.T_der= \
        exp_mod.crust_analy_params.Analytical_sol_cavity_T_Use(exp_mod.T_list.data[:exp_mod.max_count],exp_mod.P_list.data[:exp_mod.max_count],
                                                               exp_mod.radius,exp_mod.times_list.data[:exp_mod.max_count])
    IC = np.array([P_0, T_0, eps_g0, V_0, rho_m0, rho_x0])  # % store initial conditions
    exp_mod.y0 = IC
    exp_mod.perm_evl_init=np.array([])
    exp_mod.perm_evl_init_time = np.array([])
    return exp_mod,N


exp_mod,N = func_set_system()

def func_evolve_init_cond(exp_mod):
    '''
    Calculate the initial evolution of the system - regularize the pore pressure condition
    :param exp_mod:
    :return:
    '''
    ### First evolve the solution to a 1 yr (a few points is ok since everything is analytical ..)
    perm_init = exp_mod.permeability
    times_evolve_p1 = np.linspace(1e3,np.pi*1e7,10)
    for i in times_evolve_p1:
        exp_mod.P_list.update(0.)
        exp_mod.T_list.update(T_0 - exp_mod.param['T_S'])
        exp_mod.times_list.update(i)
        exp_mod.T_out, exp_mod.P_out, exp_mod.sigma_rr, exp_mod.sigma_theta, exp_mod.T_der = \
            exp_mod.crust_analy_params.Analytical_sol_cavity_T_Use(exp_mod.T_list.data[:exp_mod.max_count],
                                                                   exp_mod.P_list.data[:exp_mod.max_count],
                                                                   exp_mod.radius, exp_mod.times_list.data[:exp_mod.max_count])
        #print(i,np.max(exp_mod.P_out) / exp_mod.param['delta_Pc'])
        exp_mod.max_count += 1
    print(i,np.max(exp_mod.P_out) / exp_mod.param['delta_Pc'])
    if np.max(exp_mod.P_out) > 0.8*exp_mod.param['delta_Pc'] :
        excess_press = True
    else :
        return exp_mod

    times_evolve_p1 = np.linspace(1.5 * np.pi * 1e7, np.pi * 1e7 * 1e2, 100)
    i_count = 0
    exp_mod.perm_evl_init = np.append(exp_mod.perm_evl_init,perm_init)
    P_cond = exp_mod.P_list.data[exp_mod.max_count-1] ## Keep this constant with time for the subsequent evolution ..
    T_cond = exp_mod.T_list.data[exp_mod.max_count-1] ## Keep this constant with time for the subsequent evolution ..
    plt.figure()
    while excess_press :
        exp_mod.P_list.update(P_cond)
        exp_mod.T_list.update(T_cond)
        exp_mod.times_list.update(times_evolve_p1[i_count])
        exp_mod.T_out, exp_mod.P_out, exp_mod.sigma_rr, exp_mod.sigma_theta, exp_mod.T_der = \
            exp_mod.crust_analy_params.Analytical_sol_cavity_T_Use(exp_mod.T_list.data[:exp_mod.max_count],
                                                                   exp_mod.P_list.data[:exp_mod.max_count],
                                                                   exp_mod.radius, exp_mod.times_list.data[:exp_mod.max_count])
        exp_mod.max_count += 1
        i_count += 1
        plt.plot(exp_mod.R_outside,exp_mod.T_out)
        plt.pause(.2)
        if np.max(exp_mod.P_out) < 0.8*exp_mod.param['delta_Pc'] :
            excess_press = False
        exp_mod.permeability = exp_mod.permeability*1.25
        exp_mod.set_params_crust_calcs('Westerly_Granite')
        exp_mod.crust_analy_params.set_misc_grids(exp_mod.R_outside)
        exp_mod.perm_evl_init = np.append(exp_mod.perm_evl_init,exp_mod.permeability)
    exp_mod.perm_evl_init_time = times_evolve_p1[0:i_count-1]
    exp_mod.permeability = perm_init
    return exp_mod

func_evolve_init_cond(exp_mod)
exp_mod.times_list.finalize()
plt.ion()
plt.show()
plt.figure(10)
X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
plt.contourf(X,Y/3e7,(exp_mod.P_out_all/1e6),20,cmap='coolwarm')
plt.colorbar()


#plt.savefig(pref_val+'P_fl.pdf')
#
plt.figure(11)
X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
plt.contourf(X,Y/3e7,-(exp_mod.sigma_rr_all/1e6),20,cmap='coolwarm')
plt.colorbar()
# plt.savefig(pref_val+'sigma_rr.pdf')
#
plt.figure(12)
X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
plt.contourf(X,Y/3e7,exp_mod.T_out_all,20,cmap='coolwarm')
plt.colorbar()

plt.figure(13)
X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
plt.contourf(X,Y/3e7,-(exp_mod.sigma_eff_rr_all/1e6),20,cmap='coolwarm')
plt.colorbar()
# plt.savefig(pref_val+'sigma_rr_eff.pdf')

