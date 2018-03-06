from crystal_fraction import crystal_fraction
from exsolve import exsolve
from eos_g import eos_g
import numpy as np
import scipy
import sys

## Gas (eps_g = zero), eps_x is zero, too many crystals, 50 % crystallinity,eruption (yes/no)
#sw = [False,False,False,False,False]
def stopChamber(t,y,sw):
    # Local Variables: direction, value2, P_crit, isterminal, eruption, Q_out, value, P, value4, value1c, T, value1a, y, value3, eps_g, P_0, value1b
    # Function calls: disp, eps_x, stopChamber, isnan
    P = y[0]
    T = y[1]
    eps_g = y[2]
    P_0 = 200e6
    P_crit = 20e6
    value1a = eps_g  #% Detect eps_g approaching 0
    eps_x, tmp1,tmp2= crystal_fraction(T,eps_g)
    value1b = eps_x
    value1c = eps_x/(1.-eps_g)-0.8 # 80% crystals in magma crystal mixture ..
    value2 = eps_x-0.5
    if sw[4] : # is True (eruption)
        value3 = P_0-P
    else : # no eruption
        value3 = (P-P_0)-P_crit
    value = np.array([value1a, value1b, value1c,value2,value3])
    #print('heress')
    #isterminal = np.array([1, 1, 1, 1, 1,1]) #% Stop the integration
    #direction = np.array([0, 0, 0, 1, 1, 0])
    return value

#Helper function for handle_event
def event_switch(solver, event_info):
    """
    Turns the switches.
    """
    for i in range(len(event_info)): #Loop across all event functions
        if event_info[i] != 0:
            solver.sw[i] = not solver.sw[i] #Turn the switch

def handle_event(solver, event_info):
    """
    Event handling. This functions is called when Assimulo finds an event as
    specified by the event functions.
    """
    event_info = event_info[0]   #We only look at the state events information.
    while True:  #Event Iteration
        event_switch(solver, event_info) #Turns the switches
        b_mode = stopChamber(solver.t, solver.y, solver.sw)
        init_mode(solver) #Pass in the solver to the problem specified init_mode
        a_mode = stopChamber(solver.t, solver.y, solver.sw)
        event_info = check_eIter(b_mode, a_mode)
        #print(event_info)
        if not True in event_info: #sys.exit()s the iteration loop
            break

def init_mode(solver):
    """
    Initialize the DAE with the new conditions.
    """
    ## No change in the initial conditions (i.e. the values of the parameters when the eruption initiates .. - like P,V, ... T)
    ## Maybe can use it to switch pore-pressure degassing on/off during eruption
    #solver.y[1] = (-1.0 if solver.sw[1] else 3.0)
    #solver.y[2] = (0.0 if solver.sw[2] else 2.0)
    ## Gas (eps_g = zero), eps_x is zero, too many crystals, 50 % crystallinity,eruption (yes/no)
    if (solver.sw[3] ==True) and (solver.sw[4] == True):
        print('critical pressure reached but eps_x>0.5.')
        sys.exit(solver.t)

    if True in solver.sw[0:4] :
        print('Reached the end of the calculations since : ')
        if solver.sw[0] :
            print('eps_g became 0.')
        elif solver.sw[1] :
            print('eps_x became 0.')
        elif solver.sw[2] :
            print('eps_x/(1-eps_g) became 0.8')
        elif solver.sw[3] :
            print('eps_x became 0.5')
        sys.exit(solver.t)
        #solver.t0 = t_final*(0.9)
    #return 0

#Helper function for handle_event
def check_eIter(before, after):
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
