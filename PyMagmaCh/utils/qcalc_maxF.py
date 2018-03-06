import numpy as np
from PyMagmaCh.utils import constants as const

def qf_calc_maxF(t,qf_timescale,factor=3,maxF=0.1) :
    # Calculate q, the vertical melt flux/second, and F, the degree of
    # melting
    F_factor=1./20.*(maxF/0.15)
    # t and qf_timescale should both be in years.
    mu=2.2*qf_timescale;
    sigma=qf_timescale;

    #q=6000*normpdf(t,mu,sigma); % assume a total vertical thickness of lava of 6000 m.
    #q=6000*factor*normpdf(t,mu,sigma); % assume a total vertical thickness of lava of 6000 m., and assume 2:1 I:E ratio
    #q=6000*factor*wblpdf(t+qf_timescale/20,mu,1.5);
    q=6000*factor*wblpdf(t+(qf_timescale)**.8,mu,1.5);
    q=q/const.seconds_per_year  #% q was in units of m/year. convert to meters/sec.

    rng('shuffle')
    x=-4*pi+t/qf_timescale*4*pi;
    #%     F=1/20*(atan(x)+atan(6*pi)); %max F=0.2
    #% F=1/30*(atan(x)+atan(6*pi)); % max F=0.1
    F= F_factor*(atan(x)+atan(6*pi)); % max F set by user

     if t>qf_timescale
        dorand=rand;
        if dorand>(0.94+(0.04*(t-qf_timescale)/1e6))
            F=0.15*rand;
            if F<0.01
                F=0.01;
            q=.05*rand;
            q=q/(365*24*3600); % q was in units of m/year. convert to meters/sec.

     if F<0.005
                F=0.005;
    return q,F
