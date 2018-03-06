from scipy import sparse
import numpy as np
from scipy.linalg import solve_banded
from PyMagmaCh.process.process import Process

class Diffusion_2D(TimeDependentProcess):
    '''Parent class for implicit diffusion modules.
    Solves the 1D heat equation
    \rho C_p dT/dt = d/dx( K * dT/dx )

    The thermal conductivity K, density \rho and heat capacity Cp are in
    units  - W/m/K, kg/m^3, and J/K/kg.

    Assume that the boundary conditions are fixed temp ..
    (fix temp at the base and the top boundary ..) - need to specify
    self.param['T_base'],self.param['T_top'] :
    Note that the base and top are base and top of the grid

    Requirements : Can have only a single domain, also a single state variable
    (the diffusing field e.g. Temperature).
    Pass the inputs for temp evolution as a dict of diagnostics
    Eg. for temperature .. -- > give the k,rho,C_p as diagnostics field
    while pass as a dict :
        self.param['timestep'],
    self.param['timestep'],self.param['T_base'],self.param['T_top']

    Input flag use_banded_solver sets whether to use
        scipy.linalg.solve_banded
        rather than the default
        numpy.linalg.solve

        banded solver is faster but only works for 1D diffusion.
    Also note that the boundry condition is assumed to be Dirichlet type boundary condition ..

    '''
    def __init__(self,use_banded_solver=False,**kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.time_type = 'implicit'
        self.use_banded_solver = use_banded_solver
        for dom in self.domains.values():
            delta = np.mean(dom.axes[self.diffusion_axis].delta)
            # Note that the shape of delta = 1 - points.shape, 2 - bounds.shape
        self._make_diffusion_matrix(delta)

    def _make_diffusion_matrix(self,delta):
        '''Make the array for implicit solution of the 1D heat eqn
        - Allowed variable shaped grid +
        variable thermal conductivity, density, heat capacity
        - Implicit solving of 2D temperature equation:
        - RHO*Cp*dT/dt=d(k*dT/dx)/dx+d(k*dT/dy)/dy
        - Composing matrix of coefficients L() and vector (column) of right parts R()
        '''
        xnum = delta.shape[0]
        ynum = delta.shape[1]
        # Matrix of coefficients initialization for implicit solving
        L = sparse.csr_matrix(xnum*ynum,xnum*ynum)
        # Vector of right part initialization for implicit solving
        R = np.zeros(xnum*ynum,1);
        ## Upper boundary
        L[0:xnum,0:xnum]      =   1;
        R[0:xnum,1]           =   self.param['tback']
        ## Upper boundary
        L[0:ynum,0:ynum]      =   1;
        R[0:ynum,1]           =   self.param['tback']


        J = delta.size[0] # Size of the delta
        k_val = np.array(self.diagnostics['k']) # should be same shape as points
        rho_val = np.array(self.diagnostics['rho_c']) # should be same shape as points
        Cp_val = np.array(self.diagnostics['C_p']) # should be same shape as points

        term1a = (k_val[1:-1] + k_val[:-2])
        term1b = (k_val[1:-1] + k_val[2:])
        term3 = rho_val[1:-1]*Cp_val[1:-1]/self.param['timestep']
        term4 = delta[1:] + delta[:-1] # is same shape as k_val ..
        term5a = delta[:-1]*term4
        term5b = delta[1:]*term4
        Ka1 = (term1a/term3)/term5a
        Ka3 = (term1b/term3)/term5b
        Ka2 = Ka1 + Ka2
        add_t0 = Ka1[0]
        add_tn = Ka3[-1]
        #  Build the full banded matrix
        A = (np.diag(1. + Ka2, k=0) +
             np.diag(-Ka3[0:J-1], k=1) +
             np.diag(-Ka1[1:J], k=-1))
        self.diffTriDiag =  A
        self.add_t0 =  add_t0
        self.add_tn =  add_tn

    def _solve_implicit_banded(self,current, banded_matrix):
        #  can improve performance by storing the banded form once and not
        #  recalculating it...
        J = banded_matrix.shape[0]
        diag = np.zeros((3, J))
        diag[1, :] = np.diag(banded_matrix, k=0)
        diag[0, 1:] = np.diag(banded_matrix, k=1)
        diag[2, :-1] = np.diag(banded_matrix, k=-1)
        return solve_banded((1, 1), diag, current)

    def _implicit_solver(self):
        # Time-stepping the diffusion is just inverting this matrix problem:
        # self.T = np.linalg.solve( self.diffTriDiag, Trad )
        # Note that there should be only a single state variable - the field that is diffusing ..
        newstate = {}
        for varname, value in self.state.iteritems():
            if self.use_banded_solver:
                new_val = value[1:-1].copy()
                new_val[0] += self.param['T_base']*self.add_t0
                new_val[-1] += self.param['T_top']*self.add_tn
                newvar = self._solve_implicit_banded(new_val, self.diffTriDiag)
            else:
                new_val = value[1:-1].copy()
                new_val[0] += self.param['T_base']*self.add_t0
                new_val[-1] += self.param['T_top']*self.add_tn
                newvar = np.linalg.solve(self.diffTriDiag, new_val)
            newstate[varname][1:-1] = newvar
        return newstate

    def compute(self):
        # Time-stepping the diffusion is just inverting this matrix problem:
        # self.T = np.linalg.solve( self.diffTriDiag, Trad )
        # Note that there should be only a single state variable - the field that is diffusing ..
        newstate = self._implicit_solver()
        for varname, value in self.state.items():
            self.adjustment[varname] = newstate[varname] - value











import numpy
from scipy.linalg import solve

def constructMatrix(nx, ny, sigma):
    """ Generate implicit matrix for 2D heat equation with
        Dirichlet in bottom and right and Neumann in top and left
        Assumes dx = dy

    Parameters:
    ----------
    nx   : int
        number of discretization points in x
    ny   : int
        number of discretization points in y
    sigma: float
        alpha*dt/dx

    Returns:
    -------
    A: 2D array of floats
        Matrix of implicit 2D heat equation
    """

    A = numpy.zeros(((nx-2)*(ny-2),(nx-2)*(ny-2)))

    row_number = 0 # row counter
    for j in range(1,ny-1):
        for i in range(1,nx-1):

            # Corners
            if i==1 and j==1: # Bottom left corner (Dirichlet down and left)
                A[row_number,row_number] = 1/sigma+4 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number+nx-2] = -1   # fetch j+1

            elif i==nx-2 and j==1: # Bottom right corner (Dirichlet down, Neumann right)
                A[row_number,row_number] = 1/sigma+3 # Set diagonal
                A[row_number,row_number-1] = -1      # Fetch i-1
                A[row_number,row_number+nx-2] = -1   # fetch j+1

            elif i==1 and j==ny-2: # Top left corner (Neumann up, Dirichlet left)
                A[row_number,row_number] = 1/sigma+3   # Set diagonal
                A[row_number,row_number+1] = -1        # fetch i+1
                A[row_number,row_number-(nx-2)] = -1   # fetch j-1

            elif i==nx-2 and j==ny-2: # Top right corner (Neumann up and right)
                A[row_number,row_number] = 1/sigma+2   # Set diagonal
                A[row_number,row_number-1] = -1        # Fetch i-1
                A[row_number,row_number-(nx-2)] = -1   # fetch j-1

            # Sides
            elif i==1: # Left boundary (Dirichlet)
                A[row_number,row_number] = 1/sigma+4 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number+nx-2] = -1   # fetch j+1
                A[row_number,row_number-(nx-2)] = -1 # fetch j-1

            elif i==nx-2: # Right boundary (Neumann)
                A[row_number,row_number] = 1/sigma+3 # Set diagonal
                A[row_number,row_number-1] = -1      # Fetch i-1
                A[row_number,row_number+nx-2] = -1   # fetch j+1
                A[row_number,row_number-(nx-2)] = -1 # fetch j-1

            elif j==1: # Bottom boundary (Dirichlet)
                A[row_number,row_number] = 1/sigma+4 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number-1] = -1      # fetch i-1
                A[row_number,row_number+nx-2] = -1   # fetch j+1

            elif j==ny-2: # Top boundary (Neumann)
                A[row_number,row_number] = 1/sigma+3 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number-1] = -1      # fetch i-1
                A[row_number,row_number-(nx-2)] = -1 # fetch j-1

            # Interior points
            else:
                A[row_number,row_number] = 1/sigma+4 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number-1] = -1      # fetch i-1
                A[row_number,row_number+nx-2] = -1   # fetch j+1
                A[row_number,row_number-(nx-2)] = -1 # fetch j-1

            row_number += 1 # Jump to next row of the matrix!
    return A

def generateRHS(nx, ny, sigma, T, T_bc):
    """ Generates right-hand side for 2D implicit heat equation with Dirichlet in bottom and left and Neumann in top and right
        Assumes dx=dy, Neumann BCs = 0, and constant Dirichlet BCs

        Paramenters:
        -----------
        nx   : int
            number of discretization points in x
        ny   : int
            number of discretization points in y
        sigma: float
            alpha*dt/dx
        T    : array of float
            Temperature in current time step
        T_bc : float
            Temperature in Dirichlet BC

        Returns:
        -------
        RHS  : array of float
            Right hand side of 2D implicit heat equation
    """
    RHS = numpy.zeros((nx-2)*(ny-2))

    row_number = 0 # row counter
    for j in range(1,ny-1):
        for i in range(1,nx-1):

            # Corners
            if i==1 and j==1: # Bottom left corner (Dirichlet down and left)
                RHS[row_number] = T[j,i]*1/sigma + 2*T_bc

            elif i==nx-2 and j==1: # Bottom right corner (Dirichlet down, Neumann right)
                RHS[row_number] = T[j,i]*1/sigma + T_bc

            elif i==1 and j==ny-2: # Top left corner (Neumann up, Dirichlet left)
                RHS[row_number] = T[j,i]*1/sigma + T_bc

            elif i==nx-2 and j==ny-2: # Top right corner (Neumann up and right)
                RHS[row_number] = T[j,i]*1/sigma

            # Sides
            elif i==1: # Left boundary (Dirichlet)
                RHS[row_number] = T[j,i]*1/sigma + T_bc

            elif i==nx-2: # Right boundary (Neumann)
                RHS[row_number] = T[j,i]*1/sigma

            elif j==1: # Bottom boundary (Dirichlet)
                RHS[row_number] = T[j,i]*1/sigma + T_bc

            elif j==ny-2: # Top boundary (Neumann)
                RHS[row_number] = T[j,i]*1/sigma

            # Interior points
            else:
                RHS[row_number] = T[j,i]*1/sigma

            row_number += 1 # Jump to next row!

    return RHS

def map_1Dto2D(nx, ny, T_1D, T_bc):
    """ Takes temperatures of solution of linear system, stored in 1D,
    and puts them in a 2D array with the BCs
    Valid for constant Dirichlet bottom and left, and Neumann with zero
    flux top and right

    Parameters:
    ----------
        nx  : int
            number of nodes in x direction
        ny  : int
            number of nodes in y direction
        T_1D: array of floats
            solution of linear system
        T_bc: float
            Dirichlet BC

    Returns:
    -------
        T: 2D array of float
            Temperature stored in 2D array with BCs
    """
    T = numpy.zeros((ny,nx))

    row_number = 0
    for j in range(1,ny-1):
        for i in range(1,nx-1):
            T[j,i] = T_1D[row_number]
            row_number += 1
    # Dirichlet BC
    T[0,:] = T_bc
    T[:,0] = T_bc
    #Neumann BC
    T[-1,:] = T[-2,:]
    T[:,-1] = T[:,-2]

    return T

def btcs_2D(T, A, nt, sigma, T_bc, nx, ny, dt):
    """ Advances diffusion equation in time with backward Euler

    Parameters:
    ----------
    T: 2D array of float
        initial temperature profile
    A: 2D array of float
        Matrix with discretized diffusion equation
    nt: int
        number of time steps
    sigma: float
        alpha*dt/dx^2
    T_bc : float
        Dirichlet BC temperature
    nx   : int
        Discretization points in x
    ny   : int
        Discretization points in y
    dt   : float
        Time step size

    Returns:
    -------
    T: 2D array of floats
        temperature profile after nt time steps
    """

    j_mid = int((numpy.shape(T)[0])/2)
    i_mid = int((numpy.shape(T)[1])/2)

    for t in range(nt):
        Tn = T.copy()
        b = generateRHS(nx, ny, sigma, Tn, T_bc)
        # Use numpy.linalg.solve
        T_interior = solve(A,b)
        T = map_1Dto2D(nx, ny, T_interior, T_bc)

        # Check if we reached T=70C
        if T[j_mid, i_mid] >= 70:
            print ("Center of plate reached 70C at time {0:.2f}s, in time step {1:d}.".format(dt*t, t))
            break

    if T[j_mid, i_mid]<70:
        print ("Center has not reached 70C yet, it is only {0:.2f}C.".format(T[j_mid, i_mid]))

    return T



alpha = 1e-4

L = 1.0e-2
H = 1.0e-2

nx = 21
ny = 21
nt = 300

dx = L/(nx-1)
dy = H/(ny-1)

x = numpy.linspace(0,L,nx)
y = numpy.linspace(0,H,ny)

T_bc = 100

Ti = numpy.ones((ny, nx))*20
Ti[0,:]= T_bc
Ti[:,0] = T_bc
sigma = 0.25
A = constructMatrix(nx, ny, sigma)
dt = sigma * min(dx, dy)**2 / alpha
T = btcs_2D(Ti.copy(), A, nt, sigma, T_bc, nx, ny, dt)
