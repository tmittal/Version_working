'''input_functions.py

A collection of function definitions to handle common
calcualtions (i.e. constants and melting curve, density parameterizations)

'''
import pdb
import numpy as np
import constants as const
import warnings

class append_me:
    def __init__(self):
        self.data = np.empty((100,))
        self.capacity = 100
        self.size = 0

    def update(self, row):
        #for r in row:
        self.add(row)

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.empty((self.capacity,))
            newdata[:self.size] = self.data
            self.data = newdata
        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        self.data = self.data[:self.size]

class Parameters(object):
    def __init__(self, source):
        self.source = source

class Input_functions(object):
    def __init__(self,crust_density=None,model_source=None):
        self.crust_density = crust_density # kg/m^3
        self.model_source = model_source
    def material_constants(self):
        raise NotImplementedError('must implement a material_constants method')
    def gas_density(self,T,P):
        raise NotImplementedError('must implement a gas_density method')
    def melting_curve(self,T,P,eps_g):
        raise NotImplementedError('must implement a melting_curve method')
    def solubulity_curve(self,T,P):
        raise NotImplementedError('must implement a solubulity_curve method')
    def crit_outflow(self,*args,additional_model=None):
        raise NotImplementedError('must implement a crit_outflow method')
    def crustal_viscosity(self,T,r_val):
        raise NotImplementedError('must implement a crustal_viscosity method')
    def func_Uog(self,eps_g,eps_x,m_eq,rho_m,rho_g,T,delta_P_grad):
        raise NotImplementedError('must implement a func_Uog method')

class Input_functions_Degruyer(Input_functions):
    '''
    Extends the Input_functions to use the parameterizations
    from Degruyer & Huber 2014
    '''
    def __init__(self):
        crust_density = 2600.0 # kg/m^3
        super(Input_functions_Degruyer, self).__init__(crust_density = crust_density,model_source='Degruyter_Huber_2014')
        ## Parameters for the melting curve calculations
        # b is an exponent to approximate composition (1 = mafic, 0.5 = silicic)
        self.b = 0.5
        self.T_s = 973.0 # Kelvin , other value = 850+273.0-200
        # T_s is solidus temperature in Kelvin (Default value = 973 K)
        self.T_l = 1223.0 # Kelvin, other value = 1473.0-200.
        # T_l is liquidus temperature in Kelvin (Default value = 1223 K)
        self.Pc = 20.0 # assume a critical overpressure of 20 MPa
        self.eta_crit_lock = 0.5 # based on cystal locking above 50 % packing ..
        self.M_out_rate = 1e4 # kg/s
        self.psi_m = 0.637 #the maximum random close packing fraction for mono-sized spherical particle
        self.r_b = 100*1e-6 # radius of the bubble, in m
        self.material_constants()
        self.outflow_model = 'huppert'

    def material_constants(self):
        """
        Specify the material constants used in the paper -
        Output as a dictionary ..
        alpha_m = melt thermal expansion coefficient (1/K)
        alpha_x = crystal thermal expansion coefficient (1/K)
        alpha_r = crust thermal expansion coefficient (1/K)
        beta_x = melt bulk modulus (Pa)
        beta_m = crystal bulk modulus (Pa)
        beta_r = crust bulk modulus (Pa)
        k_crust = thermal conductivity of the crust (J/s/m/K)
        c_x,c_g,c_m = specific heat capacities (J/kg/K)
        L_m,L_e = latent heat of melting and exsolution (J/kg)
        kappa = thermal diffusivity of the crust
        """
        mat_const = {'crustal_density':self.crust_density,'beta_m': 1e10, 'alpha_m': 1e-5, 'beta_x': 1e10, 'alpha_x':1e-5, 'beta_r': 1e10, 'alpha_r':1e-5,
                'k_crust': 3.25,'c_m' : 1315.0,'c_x' : 1205.0,'c_g' : 3880.0,'L_m':290e3,'L_e':610e3,'kappa':1e-6}
        return mat_const

    def gas_density(self,T,P):
        """Compute equation of state of the gas phase.

        Input:  T is temperature in Kelvin ( 873 < T < 1173 K )
                P is pressure in Pa (30 Mpa < P < 400 MPa)
        Output: rhog, drhog_dP, drhog_dT (gas density, d(rho_g)/dP and d(rho_g)/dT)
        """
        rho_g        = -112.528*(T-273.15)**-0.381 + 127.811*(P*1e-5)**-1.135 + 112.04*(T-273.15)**-0.411*(P*1e-5)**0.033
        drho_g_dP    = (-1.135)*127.811*(P*1e-5)**-2.135 + 0.033*112.04*(T-273.15)**-0.411*(P*1e-5)**-0.967
        drho_g_dT    = (-0.381)*(-112.528)*(T-273.15)**-1.381 + (-0.411)*112.04*(T-273.15)**-1.411*(P*1e-5)**0.033
        rho_g        = rho_g*1e3
        drho_g_dP    = drho_g_dP*1e-2
        drho_g_dT    = drho_g_dT*1e3
        return rho_g,drho_g_dP,drho_g_dT

    def melting_curve(self,T,P,eps_g):
        """Compute melt fraction-temperature relationship.
        Input:  T is temperature in Kelvin
                eps_g is gas volume fraction
        Output: eta_x,deta_x_dT,deta_x_deta_g (eta_x is crystal volume fraction, others are its derivative with T and eta_g)
        """
        temp1 = T - self.T_s
        temp2 = self.T_l - self.T_s
        phi_x = 1.-  (temp1/temp2)**self.b
        dphi_x_dT =  - self.b*temp1**(self.b-1.)/(temp2)**self.b
        if T<self.T_s:
            phi_x = 1.
            dphi_x_dT = 0.
        elif T > self.T_l:
            phi_x = 0.
            dphi_x_dT = 0.
        eps_x = np.dot(1.-eps_g, phi_x)
        deps_x_dT = np.dot(1.-eps_g, dphi_x_dT)
        deps_x_deps_g = -phi_x
        return eps_x, deps_x_dT, deps_x_deps_g

    def solubulity_curve(self,T,P):
        """Compute solubility - dissolved water content in the melt
        Input:  T is temperature in Kelvin ( 873 < T < 1173 K )
                P is pressure in Pa (30 Mpa < P < 400 MPa)
        Output: meq,dmeq_dT,dmeq_dP (meq is dissolved water content others are its derivative with T and eta_g)
        """
        meq    =   (P*1e-6)**0.5*(0.4874 - 608./T + 489530./T**2.) + (P*1e-6)*(-0.06062 + 135.6/T - 69200./T**2.) + (P*1e-6)**1.5*(0.00253 - 4.154/T + 1509./T**2.)
        dmeqdP = 0.5*(P*1e-6)**-0.5*(0.4874 - 608./T + 489530./T**2.) + (-0.06062 + 135.6/T - 69200./T**2.) \
              + 1.5*(P*1e-6)**0.5*(0.00253 - 4.154/T + 1509./T**2.)
        dmeqdT =   (P*1e-6)**0.5*( 608./T**2. - 2.*489530./T**3.) \
              + (P*1e-6)*(-135.6/T**2. + 2.*69200./T**3.) \
              + (P*1e-6)**1.5*(4.154/T**2. - 2.*1509./T**3.)
        meq     = 1e-2*meq
        dmeqdP  = 1e-8*dmeqdP
        dmeqdT  = 1e-2*dmeqdT
        return meq,dmeqdP,dmeqdT

    def crit_outflow(self,*args,additional_model=None):
        """
        Specify the conditions for eruptions according to Degruyter 2014 model
        Pc = critical overpressure
        eta_x = crystal volume fraction
        M_out_rate is the mass outflow rate
        """
        if (additional_model == None) :
            M_out_rate = self.M_out_rate # kg/s
        elif additional_model == 'huppert' :
            M_out_rate = self.huppert_outflow(*args) # kg/s
        else :
            raise NotImplementedError('Not implemented this outflow method')
        return M_out_rate

    def huppert_outflow(self,eps_x,m_eq, T, rho, depth, Area_conduit,S,delta_P) :
        """
        Huppert and Woods 2003 - Eqn 7 : Q = (rho*S*(area)^2/H/mu)*delta_P
        Area_conduit = 10.*10. # 100 m^2 area ..
        S = 0.1 # shape factor ..
        used the formulation of viscosity of the melt/crystal mixture as
        described in Hess and Dingwell [1996], Parmigiani et al. 2017
        """
        mu_star = ((self.psi_m - eps_x)/(self.psi_m - self.psi_m*eps_x))**(-2.5*self.psi_m/(1.-self.psi_m))
        mu_m = 10.**( -3.545 + 0.833*np.log(100.*m_eq) +(9601. - 2368.*np.log(100.*m_eq) )/(T - (195.7 + 32.35*np.log(100*m_eq))))
        mu_mixture = mu_star*mu_m
        scale_fac = (S*(Area_conduit)**2/mu_mixture)*(rho/depth)
        return scale_fac*delta_P

    def crustal_viscosity(self,T,r_val):
        """Compute the viscosity of the visco-elastic shell surrounding the magma chamber.
        Input:  T is temperature in Kelvin, r_val is in m
        Output:
        """
        ## Parameters for the Arrhenius law :
        A = 4.25e7 #Pa s
        G = 141e3  # J/mol, Activation energy for creep
        B = 8.31 # molar gas constan, J/mol/K
        dr = np.diff(r_val)
        eta_T = np.copy(T)*0.0 + 5e21 ## Base viscosity of the crust, Pa-s
        eta_T[T>100] = A*np.exp(G/B/T[T>100])
        eta_T[eta_T>5e21] = 5e21
        integrand = 4.*np.pi*np.sum(eta_T[:-1]*r_val[:-1]*r_val[:-1]*dr)  # this is f(du)*r^2 dr over the full range
        volume_shell = (4.*np.pi/3.)*(r_val[-1]**3. - r_val[0]**3.)
        eta_effective =  integrand/volume_shell
        return eta_effective

    def func_Uog(self,eps_g,eps_x,m_eq,rho_m,rho_g,T,delta_P_grad):
        '''
        This function uses the formulation from the Parmigiani et al. 2017 paper
        to calculate the volatile flux out of the magmatic system -
        Note the correction in the relative permeability function from the original paper
        (sign needed to be corrected)
        :param eps_g:
        :param eps_x:
        :param m_eq:
        :param rho_m:
        :param rho_g:
        :param T:
        :param delta_P_grad: pressure gradient driving the flow
        '''
        #pdb.set_trace()
        if eps_g >0.5 : # Too high gas fraction ..
            return 0
        if eps_x < 0.4 : # Bubbles ...
            Y_val = 0.45 #eometrical constant derived  from data
            term1 = eps_g/self.psi_m
            U_star = (1. - Y_val*term1**(1./3.))*((1. - eps_g)/(1. - 0.5*term1))*((self.psi_m - eps_g)/(self.psi_m-self.psi_m*eps_g))**(self.psi_m/(1. - self.psi_m))
            mu_star = ((self.psi_m - eps_x)/(self.psi_m - self.psi_m*eps_x))**(-2.5*self.psi_m/(1.-self.psi_m))
            mu_m = 10.**( -3.545 + 0.833*np.log(100.*m_eq) +(9601. - 2368.*np.log(100.*m_eq) )/(T - (195.7 + 32.35*np.log(100*m_eq))))
            U_og = U_star*(rho_m - rho_g)*const.g_earth*self.r_b**2./(3.*mu_m*mu_star)
            return U_og
        visc_gas =  2.414*1e-5*(10.**(247.8/(T-140))) #;% - from Rabinowicz 1998/Eldursi EPSL 2009
        eps_g_crit = 2.75*eps_x**3. - 2.79*eps_x**2. +0.6345*eps_x+ 0.0997
        if eps_g_crit <= 0. :
            raise ValueError('eps_g_crit must not be less than zero, something is wrong')
        if (eps_x >= 0.4) and (eps_g < eps_g_crit):
            return 0
        if (eps_x >= 0.4) and (eps_x <= 0.7) and  (eps_g > eps_g_crit):
            k = 1e-4*(-0.0534*eps_x**3. + 0.1083*eps_x**2. - 0.0747*eps_x + 0.0176) # m^2
            k_rel = -2.1778*eps_x**4. + 5.1511*eps_x**3. - 4.5199*eps_x**2. + 1.7385*eps_x - 0.2461
            if eps_g < eps_g_crit + 0.04 :
                f_s = ((eps_g - eps_g_crit)/0.04)**4.
            else :
                f_s = 1.
            U_og = (f_s*k*k_rel/visc_gas)*(delta_P_grad + const.g_earth*(rho_m-rho_g))
            if U_og/1e-3 > 1:  # This is to ensure an upper limit cutoff ..
                warnings.warn('Exceeded upper limit of fluid velocity - 1e-3',UserWarning)
                U_og = 1e-3
            return U_og
        raise ValueError('Error - the code should not be here')

class Input_functions_DeKa(Input_functions_Degruyer):
    '''
    Extends the Input_functions to use the parameterizations
    from Degruyer & Huber 2014 + some melting things from Karlstrom 2009
    '''
    def __init__(self):
        super(Input_functions_DeKa, self).__init__()
        self.model_source='Degruyter_Huber_2014; karlstrom_2009'
        self.anhyd = True

    def melting_curve(self,T,P,eps_g):
        '''
        from Karlstrom 2009 melting curve, Anhydrous ..
        as well as the hydr .. 2% (new exp make this obselete ?? since it seems too broad)
        :param T:
        :param P:
        :param eps_g: gas fraction
        :return:eps_x, deps_x_dT, deps_x_deps_g
        '''
        if self.anhyd == True :
            T = (T-273.15) + 12.*(15. - P/100000000.)
            melt_frac = 2.79672e-11*(T**4.) - 8.79939e-8*(T**3.) + 1.01622e-4*T**2. - 5.02861e-2*T + 8.6693
            if melt_frac < 0:
                melt_frac = 0.
            elif melt_frac > 1:
                melt_frac = 1.
            phi_x = 1. - melt_frac
            eps_x = np.dot(1.-eps_g, phi_x)
            deps_x_deps_g = -phi_x
            dphi_x_dT =  - (2.79672e-11*(T**3.)*4. - 3.*8.79939e-8*(T**2.) + 1.01622e-4*2.*T - 5.02861e-2)
            deps_x_dT = np.dot(1.-eps_g, dphi_x_dT)
        else :
            T = (T - 273.15) + 12. * (15. - P / 100000000.)
            melt_frac = 2.039e-09 * (T ** 3.) - 3.07e-6 * (T ** 2.) + 1.63e-3 * T - 0.307
            if melt_frac < 0:
                melt_frac = 0.
            elif melt_frac > 1:
                melt_frac = 1.
            phi_x = 1. - melt_frac
            eps_x = np.dot(1. - eps_g, phi_x)
            deps_x_deps_g = -phi_x
            dphi_x_dT = - (2.039e-09 * (T ** 2.) * 3. - 3.07e-6 * (T * 2.) + 1.63e-3)
            deps_x_dT = np.dot(1. - eps_g, dphi_x_dT)
        return eps_x, deps_x_dT, deps_x_deps_g
