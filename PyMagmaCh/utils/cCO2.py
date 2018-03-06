def cCO2(bCO2,bH2O,f) :
    '''
    Calculate concentrations of water and CO2 in the melt for a given bulk
    composition and degree of melting
    '''
    D_h2o=0.01 # Katz, 2003
    D_co2=0.0001; # highly incompatible; cf. E.H. Hauri et al. / Earth and Planetary Science Letters 248 (2006) 715?734
    # Xmelt = Xbulk / (D + F(1-D))
    H2O=bH2O/(D_h2o + f*(1.-D_h2o));
    CO2=bCO2/(D_co2 + f*(1.-D_co2));
    return CO2,H2O
