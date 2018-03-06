import numpy as numpy
from numpy import pi
from PyMagmaCh_Single.Analytical_sol_cavity_T_Use import Analytical_sol_cavity_T_Use
import pylab as plt
import PyMagmaCh_Single.input_functions as md_deg
from PyMagmaCh_Single.plot_mainChamber import plot_mainChamber
from PyMagmaCh_Single.mainChamber_working_Final import Chamber_Problem
from assimulo.solvers import CVode


#% set the mass inflow rate
mdot           = 1.   #;          % mass inflow rate (kg/s) #% use global variable
depth          = 7841.0
with_plots     = True

##############################################
#% time
end_time       = 3e7*80000.#; % maximum simulation time in seconds
begin_time     = 0 #; % initialize time
##############################################

def func_set_system():
    ##############################################
    #% initial conditions
    P_0           = 200e6           #;      % initial chamber pressure (Pa)
    T_0           = 1200            #;       % initial chamber temperature (K)
    eps_g0        = 0.04            #;       % initial gas volume fraction
    rho_m0        = 2650            #;       % initial melt density (kg/m^3)
    rho_x0        = 3065            #;       % initial crystal density (kg/m^3)
    a             = 1000            #;       % initial radius of the chamber (m)
    V_0           = (4.*pi/3.)*a**3.  #; % initial volume of the chamber (m^3)

    ##############################################
    ##############################################
    IC = numpy.array([P_0,T_0,eps_g0,V_0,rho_m0,rho_x0]) #   % store initial conditions
    ## Gas (eps_g = zero), eps_x is zero, too many crystals, 50 % crystallinity,eruption (yes/no)

    sw0 = [False,False,False,False,False]

    ##############################################
    #% error tolerances used in ode method
    dt = 30e7
    N  = int(round((end_time-begin_time)/dt))
    ##############################################

    #Define an Assimulo problem
    exp_mod = Chamber_Problem(depth=depth,t0=begin_time,y0=IC,sw0=sw0)
    exp_mod.param['Mdot_in']    = mdot
    exp_mod.param['delta_Pc']   = 20e6
    exp_mod.tcurrent = begin_time
    exp_mod.radius = a
    exp_mod.permeability = 1e-19
    exp_mod.R_steps = 1000
    exp_mod.dt_init = dt
    #################
    exp_mod.R_outside = numpy.linspace(a,3.*a,exp_mod.R_steps);
    exp_mod.T_out_all =numpy.array([exp_mod.R_outside*0.])
    exp_mod.P_out_all =numpy.array([exp_mod.R_outside*0.])
    exp_mod.sigma_rr_all    = numpy.array([exp_mod.R_outside*0.])
    exp_mod.sigma_theta_all = numpy.array([exp_mod.R_outside*0.])
    exp_mod.sigma_eff_rr_all = numpy.array([exp_mod.R_outside*0.])
    exp_mod.sigma_eff_theta_all = numpy.array([exp_mod.R_outside*0.])

    exp_mod.P_list = numpy.array([P_0-exp_mod.plith])
    exp_mod.T_list = numpy.array([T_0-exp_mod.param['T_S']])
    exp_mod.times_list = numpy.array([1e-7])
    exp_mod.T_out,exp_mod.P_out,exp_mod.sigma_rr,exp_mod.sigma_theta,exp_mod.T_der= Analytical_sol_cavity_T_Use(exp_mod.T_list,exp_mod.P_list,exp_mod.radius,exp_mod.times_list,exp_mod.R_outside,exp_mod.permeability,exp_mod.param['material'])
    exp_mod.param['heat_cond'] = 1                                             # Turn on/off heat conduction
    exp_mod.param['visc_relax'] = 1                                            # Turn on/off viscous relaxation
    exp_mod.param['press_relax'] = 0                                          ## Turn on/off pressure diffusion
    exp_mod.param['frac_rad_Temp'] =0.75
    exp_mod.param['vol_degass'] = 0.
    #exp_mod.state_events = stopChamber #Sets the state events to the problem
    #exp_mod.handle_event = handle_event #Sets the event handling to the problem
    #Sets the options to the problem
    #exp_mod.p0   = [beta_r, beta_m]#, beta_x, alpha_r, alpha_m, alpha_x, L_e, L_m, c_m, c_g, c_x, eruption, heat_cond, visc_relax]  #Initial conditions for parameters
    #exp_mod.pbar  = [beta_r, beta_m]#, beta_x, alpha_r, alpha_m, alpha_x, L_e, L_m, c_m, c_g, c_x, eruption, heat_cond, visc_relax]

    #Define an explicit solver
    exp_sim = CVode(exp_mod) #Create a CVode solver

    #Sets the parameters
    #exp_sim.iter = 'Newton'
    #exp_sim.discr = 'BDF'
    #exp_sim.inith = 1e-7

    exp_sim.rtol = 1.e-7
    exp_sim.maxh = 3e7
    exp_sim.atol = 1e-7
    exp_sim.sensmethod = 'SIMULTANEOUS' #Defines the sensitvity method used
    exp_sim.suppress_sens = False       #Dont suppress the sensitivity variables in the error test.
    #exp_sim.usesens = True
    #exp_sim.report_continuously = True
    return exp_mod,exp_sim,N

exp_mod,exp_sim,N = func_set_system()
#Simulate
t_final_new = 0.
try :
    t1, y1 = exp_sim.simulate(end_time,N) #Simulate 5 seconds
    exp_sim.print_event_data()
except SystemExit:
    print('Stop Before end_time')
    t_final_new = exp_sim.t*0.9999
    exp_mod,exp_sim,N = func_set_system()
    t1, y1 = exp_sim.simulate(t_final_new,N)

print('Final Stopping time : %.2f Yrs' % (t_final_new/(3600.*24.*365.)))


if with_plots:
        t1 = numpy.asarray(t1)
        y1 = numpy.asarray(y1)
        #IC = numpy.array([P_0,T_0,eps_g0,V_0,rho_m0,rho_x0]) #   % store initial conditions
        P = y1[:,0]
        T = y1[:,1]
        eps_g = y1[:,2]
        V = y1[:,3]
        rho_m = y1[:,4]
        rho_x = y1[:,5]
        size_matrix = numpy.shape(P)[0]

        #%crystal volume fraction
        eps_x = numpy.zeros(size_matrix)
        #% dissolved water mass fraction
        m_eq = numpy.zeros(size_matrix)
        #% gas density
        rho_g = numpy.zeros(size_matrix)

        for i in range(0,size_matrix) :
            eps_x[i],tmp1,tmp2 =  md_deg.melting_curve_degruyter(T[i],eps_g[i]);
            m_eq[i],tmp1,tmp2 =  md_deg.solubulity_curve_degruyter(T[i],P[i])
            rho_g[i],tmp1,tmp2 = md_deg.gas_density_degruyter(T[i],P[i])
        #% bulk density
        rho  = (1.-eps_g-eps_x)*rho_m + eps_g*rho_g + eps_x*rho_x
        #% bulk heat capacity
        c  = ((1-eps_g-eps_x)*rho_m*exp_mod.param['c_m'] + eps_g*rho_g*exp_mod.param['c_g'] + eps_x*rho_x*exp_mod.param['c_x'])/rho;
        plot_mainChamber(t1,V,P,T,eps_x,eps_g,rho,'no_diff_')

plt.ion()
plt.show()

plt.figure(10)
X,Y = numpy.meshgrid(exp_mod.R_outside,exp_mod.times_list)
plt.contourf(X,Y/3e7,(exp_mod.P_out_all/1e6),20,cmap='coolwarm')
plt.colorbar()
plt.savefig('No_diff_P_fl.pdf')

plt.figure(11)
X,Y = numpy.meshgrid(exp_mod.R_outside,exp_mod.times_list)
plt.contourf(X,Y/3e7,-(exp_mod.sigma_rr_all/1e6),20,cmap='coolwarm')
plt.colorbar()
plt.savefig('No_diff_sigma_rr.pdf')

plt.figure(12)
X,Y = numpy.meshgrid(exp_mod.R_outside,exp_mod.times_list)
plt.contourf(X,Y/3e7,-(exp_mod.sigma_eff_rr_all/1e6),20,cmap='coolwarm')
plt.colorbar()
plt.savefig('No_diff_sigma_rr_eff.pdf')
