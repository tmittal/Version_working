import numpy as np

def katz_wetmelting(T_kelvin, P_pascal, X,D_h2o_n = 0) :
    '''Calculates degree of melting F[wt.frac.] as a function of
    temperature T[degrees K], pressure P[Pa],  and water content X [wt.frac.]
    according to parametrization by Katz et al. (2003). Use - D_h2o_n =1
    for Dh2o = 0.012 #% Kelley et al., (2006, 2010)
    '''
    #%%%%%%%%%%%%%%%%%%%%%%% Initialization%%%%%%%%%%%%%%%%%%%%%%%%%
    P = P_pascal*1e6 # P in GPa
    T = T_kelvin - 273.15 # T in Celcius
    f = 0.0
    mcpx=0.17
    beta1 = 1.50
    beta2 = 1.50
    A1 = 1085.7
    A2 = 132.9
    A3 = -5.1
    B1 = 1475.0
    B2 = 80.0
    B3 = -3.2
    C1 = 1780.0
    C2 = 45.0
    C3 = -2.0
    r0 = 0.50 # called r1 in Katz et al. 2003 Table 2
    r1 = 0.08 # called r2 in Katz et al. 2003 Table 2
    #%%%% Wet melting parameters
    K=43. #% degrees C/wt% water
    gamma=0.75 #%wet melting exponent
    zeta1=12.00
    zeta2=1.00
    ramda=0.60
    if (D_h2o_n != 0):
        Dh2o = 0.012 #% Kelley et al., (2006, 2010) -- NOTE Katz et al. (2003) assume D=0.01
    else :
        Dh2o = 0.01
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # calculate Rcpx(P)
    r_cpx = r0 + r1*P
    #% compute F_cpx-out
    f_cpxout = mcpx/r_cpx
    #% compute liquidus temperature
    T_liquidus = C1 + C2*P + C3*P**2
    #% compute solidus temperature
    T_solidus = A1 + A2*P + A3*P**2
    #% compute lherzolite liquidus temperature
    T_lherzliq = B1 + B2*P + B3*P**2
    T_cpxout = ((f_cpxout)**(1.0/beta1))*(T_lherzliq - T_solidus) + T_solidus
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (X<=0) :
        if( T < T_solidus) :
            f=0.
        elif(T < T_cpxout) :
            Tprime = (T-T_solidus)/(T_lherzliq - T_solidus)
            f = Tprime**beta1
        elif((T >= T_cpxout) and (T < T_liquidus)) :
            f = f_cpxout + (1.0 - f_cpxout)*(( (T-T_cpxout)/(T_liquidus-T_cpxout))**beta2)
        else :
            f=1.0
        return f # Stop here for anyhdrous case ..
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    maxF=f_cpxout
    minF=0.
    f=0.
    if (X>0):  # if hydrous melting
        Xh2o_sat=zeta1*(P**ramda)+zeta2*P
        if(X>Xh2o_sat):
                X=Xh2o_sat
        Xwater=X/(Dh2o+f*(1.-Dh2o))
        deltaT=K*(Xwater**gamma)
        Xwater_cpx=X/(Dh2o+maxF*(1.-Dh2o))
        deltaTmin=K*(Xwater_cpx**gamma)
        if( T < (T_solidus-deltaT)) : #% if no melting
            return f
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        elif(T < (T_cpxout-deltaTmin)) : #% if  melting less than f_cpxout
          Xwater=X/(Dh2o+f*(1,-Dh2o))
          deltaT=K*Xwater**gamma
          fnew=((T-(T_solidus-deltaT))/(T_lherzliq - T_solidus))**beta1
          fdiff=np.abs(fnew-f)
          nloops=0
          while (fdiff>1e-7) :
              if (fnew>f) :
                  minF=f
                  f = (f+maxF)/2.
              elif (fnew<f) :
                  maxF=f
                  f = (f+minF)/2. #% Can this be narrowed down further?
              else :
                  return f
              Xwater=X/(Dh2o+f*(1,-Dh2o))
              deltaT=K*(Xwater**gamma)
              fnew=((T-(T_solidus-deltaT))/(T_lherzliq - T_solidus))**beta1
              fdiff=np.abs(fnew-f) #% check for convergence
              nloops=nloops+1
              if (nloops>100) :
                  fdiff = 0. #% prevent infinite looping if something is broken
          return f
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        elif (T < T_liquidus): # % if  melting more than f_cpxout, less than f=1
          maxF=1.
          Xwater=X/(Dh2o+f*(1-Dh2o))
          deltaT=K*Xwater**gamma
          fnew=f_cpxout + (1.0 - f_cpxout)*(( (T-(T_cpxout-deltaT))/(T_liquidus-(T_cpxout-deltaT)))**beta2)
          fdiff=np.abs(fnew-f)
          nloops=0.
          while (fdiff>1e-7) :
                if (fnew>f) :
                    minF=f
                    f = (f+maxF) / 2.
                elif (fnew<f):
                    maxF=f
                    f = (f+minF)/2. # % Can this be narrowed down further?
                else: #% Xcalc==X
                    return f
                Xwater=X/(Dh2o+f*(1-Dh2o))
                deltaT= K*Xwater**gamma
                fnew=f_cpxout + (1.0 - f_cpxout)*(( (T-(T_cpxout-deltaT))/(T_liquidus-(T_cpxout-deltaT)))**beta2)
                fdiff= np.abs(fnew-f) #% check for convergence
                nloops=nloops+1
                if (nloops>100) :
                  fdiff = 0. #% prevent infinite looping if something is broken
          return f
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else:
                f=1.0
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    return f
