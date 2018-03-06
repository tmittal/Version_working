from assimulo.solvers import CVode,LSODAR
import sys
import numpy as np
import pylab as plt
from plot_mainChamber import plot_mainChamber
from mainChamber_working_Final import Chamber_Problem
import input_functions as inp

pref_val = sys.argv[1] #'1p17_diff_'
perm_val = float(sys.argv[2])
#print(sys.argv)

#% set the mass inflow rate
mdot          = 1.   #;          % mass inflow rate (kg/s) #% use global variable
depth  =  8000.
with_plots=True

##############################################
#% time
end_time       = 3e7*5e5#; % maximum simulation time in seconds
##############################################

def func_set_system():
    ##############################################
    #% initial conditions
    P_0           = depth*9.8*2600.           #;      % initial chamber pressure (Pa)
    T_0           = 1200            #;       % initial chamber temperature (K)
    eps_g0        = 0.04            #;       % initial gas volume fraction
    rho_m0        = 2600            #;       % initial melt density (kg/m^3)
    rho_x0        = 3065            #;       % initial crystal density (kg/m^3)
    a             = 1000            #;       % initial radius of the chamber (m)
    V_0           = (4.*np.pi/3.)*a**3.  #; % initial volume of the chamber (m^3)

    ##############################################
    ##############################################
    IC = np.array([P_0, T_0, eps_g0, V_0, rho_m0, rho_x0])  # % store initial conditions
    ## Gas (eps_g = zero), eps_x is zero, too many crystals, 50 % crystallinity,eruption (yes/no)
    sw0 = [False,False,False,False,False]

    ##############################################
    #% error tolerances used in ode method
    dt = 30e7
    begin_time = 0  # ; % initialize time
    N  = int(round((end_time-begin_time)/dt))
    ##############################################

    #Define an Assimulo problem
    exp_mod = Chamber_Problem(depth=depth,t0=begin_time,y0=IC,sw0=sw0)
    exp_mod.param['T_S'] = 250.+273.
    exp_mod.param['T_in'] = 1200.
    exp_mod.param['eps_g_in'] = 0.0    # Gas fraction of incoming melt - gas phase ..
    exp_mod.param['m_eq_in'] = 0.03    # Volatile fraction of incoming melt
    exp_mod.param['Mdot_in']    = mdot
    exp_mod.param['eta_x_max'] = 0.64                                     # Locking fraction
    exp_mod.param['delta_Pc']   = 20e6
    exp_mod.tcurrent = begin_time
    exp_mod.radius = a
    exp_mod.permeability = perm_val
    exp_mod.R_steps = 1500
    exp_mod.dt_init = dt
    inp_func1 = inp.Input_functions_Degruyer()
    exp_mod.set_input_functions(inp_func1)
    exp_mod.get_constants()
    #################
    exp_mod.R_outside = np.linspace(a,3.*a,exp_mod.R_steps)
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
    exp_mod.times_list.update(1e3)
    exp_mod.T_out,exp_mod.P_out,exp_mod.sigma_rr,exp_mod.sigma_theta,exp_mod.T_der= \
        exp_mod.crust_analy_params.Analytical_sol_cavity_T_Use(exp_mod.T_list.data[:exp_mod.max_count],exp_mod.P_list.data[:exp_mod.max_count],
                                                               exp_mod.radius,exp_mod.times_list.data[:exp_mod.max_count])

    exp_mod.perm_evl_init=np.array([])
    exp_mod.perm_evl_init_time = np.array([])
    begin_time = exp_mod.func_evolve_init_cond(T_0)
    exp_mod.t0 = begin_time
    exp_mod.param['heat_cond'] = 1.                                             # Turn on/off heat conduction
    exp_mod.param['visc_relax'] = 1.                                          # Turn on/off viscous relaxation
    exp_mod.param['press_relax'] = 1.                                          ## Turn on/off pressure diffusion
    exp_mod.param['frac_rad_Temp'] =0.75
    exp_mod.param['vol_degass'] = 1.
    exp_mod.param['outflow_model'] = None  # 'huppert'
    IC = np.array([P_0, T_0, eps_g0, V_0, rho_m0, rho_x0])  # % store initial conditions
    exp_mod.y0 = IC

    #exp_mod.state_events = stopChamber #Sets the state events to the problem
    #exp_mod.handle_event = handle_event #Sets the event handling to the problem
    #Sets the options to the problem
    #exp_mod.p0   = [beta_r, beta_m]#, beta_x, alpha_r, alpha_m, alpha_x, L_e, L_m, c_m, c_g, c_x, eruption, heat_cond, visc_relax]  #Initial conditions for parameters
    #exp_mod.pbar  = [beta_r, beta_m]#, beta_x, alpha_r, alpha_m, alpha_x, L_e, L_m, c_m, c_g, c_x, eruption, heat_cond, visc_relax]
    #Define an explicit solver
    #exp_sim = CVode(exp_mod) #Create a CVode solver
    exp_sim = LSODAR(exp_mod) #Create a CVode solver

    #Sets the parameters
    exp_sim.store_event_points = True
    #exp_sim.iter = 'Newton'
    #exp_sim.discr = 'BDF'
    #exp_sim.inith = 1e-7
    #exp_sim.display_progress = True
    #exp_sim.rtol = 1.e-7
    #exp_sim.maxh = 3e7
    #exp_sim.atol = 1e-7
    #exp_sim.sensmethod = 'SIMULTANEOUS' #Defines the sensitvity method used
    #exp_sim.suppress_sens = True       #Dont suppress the sensitivity variables in the error test.
    #exp_sim.usesens = True
    #exp_sim.report_continuously = True
    return exp_mod,exp_sim,N

#########################################################
#########################################################

exp_mod,exp_sim,N = func_set_system()
#Simulate
t_final_new = 0.
try :
    t1, y1 = exp_sim.simulate(end_time,N) #Simulate 5 seconds
    exp_sim.print_event_data()
except SystemExit:
    print('Stop Before end_time')
    t1 = exp_sim.t_sol
    y1 = exp_sim.y_sol
    exp_sim.print_event_data()
    # t_final_new = exp_sim.t*0.985
    #exp_mod,exp_sim,N = func_set_system()
    #t1, y1 = exp_sim.simulate(t_final_new,N)

print('Final Stopping time : %.2f Yrs' % (t_final_new/(3600.*24.*365.)))
del exp_sim
print('Number of eruptions : {:f}'.format(exp_mod.eruption_count))
print(exp_mod.eruption_events)

if with_plots:
        t1 = np.asarray(t1)
        y1 = np.asarray(y1)
        #IC = np.array([P_0,T_0,eps_g0,V_0,rho_m0,rho_x0]) #   % store initial conditions
        P = y1[:,0]
        T = y1[:,1]
        eps_g = y1[:,2]
        V = y1[:,3]
        rho_m = y1[:,4]
        rho_x = y1[:,5]
        size_matrix = np.shape(P)[0]

        #%crystal volume fraction
        eps_x = np.zeros(size_matrix)
        #% dissolved water mass fraction
        m_eq = np.zeros(size_matrix)
        #% gas density
        rho_g = np.zeros(size_matrix)
        for i in range(0,size_matrix) :
            eps_x[i],tmp1,tmp2 =  exp_mod.input_functions.melting_curve(T[i],P[i],eps_g[i])
            m_eq[i],tmp1,tmp2 =  exp_mod.input_functions.solubulity_curve(T[i],P[i])
            rho_g[i],tmp1,tmp2 = exp_mod.input_functions.gas_density(T[i],P[i])
        #% bulk density
        rho  = (1.-eps_g-eps_x)*rho_m + eps_g*rho_g + eps_x*rho_x
        #% bulk heat capacity
        c  = ((1-eps_g-eps_x)*rho_m*exp_mod.param['c_m'] + eps_g*rho_g*exp_mod.param['c_g'] + eps_x*rho_x*exp_mod.param['c_x'])/rho;
        plot_mainChamber(t1,V,P,T,eps_x,eps_g,rho,exp_mod.param['crustal_density'],pref=pref_val)


exp_mod.times_list.finalize()

plt.ion()
plt.show()
# plt.figure(10)
# X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
# plt.contourf(X,Y/3e7,(exp_mod.P_out_all/1e6),20,cmap='coolwarm')
# plt.colorbar()
#
# #plt.savefig(pref_val+'P_fl.pdf')
# #
# plt.figure(11)
# X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
# plt.contourf(X,Y/3e7,-(exp_mod.sigma_rr_all/1e6),20,cmap='coolwarm')
# plt.colorbar()
# # plt.savefig(pref_val+'sigma_rr.pdf')
# #
# plt.figure(12)
# X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
# plt.contourf(X,Y/3e7,exp_mod.T_out_all,20,cmap='coolwarm')
# plt.colorbar()
#
# plt.figure(13)
# X,Y = np.meshgrid(exp_mod.R_outside,exp_mod.times_list.data)
# plt.contourf(X,Y/3e7,-(exp_mod.sigma_eff_rr_all/1e6),20,cmap='coolwarm')
# plt.colorbar()
# # plt.savefig(pref_val+'sigma_rr_eff.pdf')
#
exp_mod.flux_in_vol.finalize()
exp_mod.flux_out_vol.finalize()

exp_mod.flux_in_vol.data = np.delete(exp_mod.flux_in_vol.data, 0)
exp_mod.flux_out_vol.data = np.delete(exp_mod.flux_out_vol.data, 0)

plt.figure(14)
plt.plot(exp_mod.times_list.data[1+exp_mod.extra_vals:]/3.142e7,exp_mod.flux_in_vol.data,'k')
plt.plot(exp_mod.times_list.data[1+exp_mod.extra_vals:]/3.142e7,exp_mod.flux_out_vol.data,'r')
plt.show()

time_steps = np.diff(exp_mod.times_list.data[exp_mod.extra_vals:])
tmp1 = np.where(exp_mod.flux_out_vol.data<1.)
vol_flux_out_non_erupt = np.sum(exp_mod.flux_out_vol.data[tmp1]*time_steps[tmp1])

tmp2 = np.where(exp_mod.flux_out_vol.data>=1.)
vol_flux_out_erupt = np.sum(exp_mod.flux_out_vol.data[tmp2]*time_steps[tmp2])

vol_flux_in = np.sum(exp_mod.flux_in_vol.data*time_steps)

print('vol_flux_out_erupt/vol_flux_in : ',vol_flux_out_erupt/vol_flux_in)
print('vol_flux_out_non_erupt/vol_flux_in : ',vol_flux_out_non_erupt/vol_flux_in)
#del exp_mod
