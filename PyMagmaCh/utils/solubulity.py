import numpy as np

def solubility_Iacano(bCO2,bH2O,P_pascal,T,sio2,tio2,al2o3,feo,fe2o3,mgo,cao,na2o,k2o) : ## output is [CO2 H2O]
    '''
    P in Pascal, T in K!!!
    Give weight percent of different oxides
    Calculate joint solubility of water and CO2 in the melt
    Using the equations in:
    Iacono-Marziano, Giada, Yann Morizet, Emmanuel Le Trong,
    and Fabrice Gaillard. "New experimental data and semi-empirical
    parameterization of H 2 O?CO 2 solubility in mafic melts."
    Geochimica et Cosmochimica Acta 97 (2012): 1-23.
    '''
    P = P_pascal*1e-5 # pressure in bars
    #Constants (anhydrous)
    #Table 5
    dh2o=2.3
    dAI= 3.8
    dfeomgo=-16.3
    dnak=20.1
    aco2=1.0
    bco2=15.8
    Cco2=0.14
    Bco2=-5.3
    #Table 6
    ah2o=0.54
    bh2o=1.24
    Bh2o= -2.95
    Ch2o=0.02
    #Convert weight percent of different oxides to mole fractions
    msio2=sio2/60.08
    mtio2=tio2/80.0
    mal2o3=al2o3/101.96
    mfeo=feo/71.84+2.0*fe2o3/159.69
    mmgo=mgo/40.3
    mcao=cao/56.08
    mna2o=na2o/61.98
    mk2o=k2o/94.2
    mh2o=bH2O/18.0
    mTot=msio2+mtio2+mal2o3+mfeo+mmgo+mcao+mna2o+mk2o
    XK2O=mk2o/mTot
    XNa2O=mna2o/mTot
    XCaO=mcao/mTot
    XMgO=mmgo/mTot
    XFeO=mfeo/mTot
    XAl2O3=mal2o3/mTot
    XSiO2=msio2/mTot
    XTiO2=mtio2/mTot
    Xh2o=mh2o/mTot      # mole fraction in the melt
    xAI=XAl2O3/(XCaO+XK2O+XNa2O)
    xfeomgo=XFeO+XMgO
    xnak=XNa2O+XK2O
    ##########################################################
    #Calculate NBO/O (See appendix 1, Iacono-Marziano et al.)
    #X_[...] is the mole fraction of different oxides.
    NBO=2.*(XK2O + XNa2O + XCaO + XMgO + XFeO - XAl2O3)
    O=(2.*XSiO2 + 2.*XTiO2+3.*XAl2O3 + XMgO + XFeO + XCaO + XNa2O + XK2O)
    nbo_o=NBO/O

    closeenough=0.0
    xco2=0.999 #xco2=(bCO2/44e4)/(bCO2/44e4+mh2o)
    xh2o=1.-xco2  # mole fraction in the vapor
    mindiff= 0.01
    maxh2o=1.
    maxco2=1.
    minco2=0.0
    n=0.0
    while (closeenough==0) and (n<30) :
        n=n+1
        #Pco2 is total pressure * xco2
        Pco2=P*xco2
        #Ph2o is total pressure * xh2o
        Ph2o=P*xh2o
        #ppm
        lnCO2=(Xh2o*dh2o+xAI*dAI+xfeomgo*dfeomgo+xnak*dnak)+aco2*np.log(Pco2)+bco2*(nbo_o)+Bco2+Cco2*P/T
        CO2=np.exp(lnCO2)
        #wt#
        lnH2O=ah2o*np.log(Ph2o)+bh2o*nbo_o+Bh2o+Ch2o*P/T
        H2O=np.exp(lnH2O)
        vCO2=bCO2-CO2
        vH2O=bH2O-H2O
        if (vCO2<0) and (vH2O<0) :
            break
        elif (vCO2<0) :
            maxco2=xco2.copy()
            xh2o=(xh2o+maxh2o)/2.
            xco2=1.0 - xh2o
        elif (vH2O<0) :
            maxh2o=xh2o.copy()
            xco2=(xco2+maxco2)/2.
            xh2o=1.0 -xco2
        else :
            xCO2m=(CO2/44e4)/(CO2/44e4+H2O/18)
            xH2Om=1.-xCO2m
            # xco2=(xco2+xCO2m)/2
            if (xCO2m>xco2) :
                xh2o=(xh2o+maxh2o)/2.
                maxco2=xco2.copy()
                xco2=1.-xh2o
             else :
                maxco2=xco2.copy()
                xh2o=(xh2o+maxh2o)/2.
                xco2=1. -xh2o
            if (np.abs(xco2-xCO2m)<mindiff):
                closeenough = 1
    return CO2,H2O # CO2 in ppm, H2O in wt %
