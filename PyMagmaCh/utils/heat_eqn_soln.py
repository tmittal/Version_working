import numpy
from scipy.linalg import solve

## The code has some helper functions for finite difference methods for heat conduction eqn solution ..
# First explicit method ...
def ftcs_mixed(T, nt, dt, dx, alpha):
    """Solves the diffusion equation with forward-time, centered scheme using
    Dirichlet b.c. at left boundary and Neumann b.c. at right boundary

    Parameters:
    ----------
    u: array of float
        Initial temperature profile
    nt: int
        Number of time steps
    dt: float
        Time step size
    dx: float
        Mesh size
    alpha: float
        Diffusion coefficient (thermal diffusivity)

    Returns:
    -------
    u: array of float
        Temperature profile after nt time steps with forward in time scheme

    Example :
    -------
	L = 1
	nt = 100
	nx = 51
	alpha = 1.22e-3

	dx = L/(nx-1)

	Ti = numpy.zeros(nx)
	Ti[0] = 100
	sigma = .5
    dt = sigma * dx*dx/alpha
    nt = 1000
	T = ftcs_mixed(Ti.copy(), nt, dt, dx, alpha)
    """
    for n in range(nt):
        Tn = T.copy()
        T[1:-1] = Tn[1:-1] + alpha*dt/dx**2*(Tn[2:] -2*Tn[1:-1] + Tn[0:-2])
        T[-1] = T[-2]
    return T


########################## Implicit method
def generateMatrix(N, sigma):
    """ Computes the matrix for the diffusion equation with backward Euler
        Dirichlet condition at i=0, Neumann at i=-1

    Parameters:
    ----------
    T: array of float
        Temperature at current time step
    sigma: float
        alpha*dt/dx^2

    Returns:
    -------
    A: 2D numpy array of float
        Matrix for diffusion equation
    """

    # Setup the diagonal
    d = numpy.diag(numpy.ones(N-2)*(2+1./sigma))

    # Consider Neumann BC
    d[-1,-1] = 1+1./sigma

    # Setup upper diagonal
    ud = numpy.diag(numpy.ones(N-3)*-1, 1)

    # Setup lower diagonal
    ld = numpy.diag(numpy.ones(N-3)*-1, -1)

    A = d + ud + ld

    return A

def generateRHS(T, sigma, qdx):
    """ Computes right-hand side of linear system for diffusion equation
        with backward Euler

    Parameters:
    ----------
    T: array of float
        Temperature at current time step
    sigma: float
        alpha*dt/dx^2
    qdx: float
        flux at right boundary * dx

    Returns:
    -------
    b: array of float
        Right-hand side of diffusion equation with backward Euler
    """

    b = T[1:-1]*1./sigma
    # Consider Dirichlet BC
    b[0] += T[0]
    # Consider Neumann BC
    b[-1] += qdx

    return b

def implicit_ftcs(T, A, nt, sigma, qdx):
    """ Advances diffusion equation in time with implicit central scheme

    Parameters:
    ----------
    T: array of float
        initial temperature profile
    A: 2D array of float
        Matrix with discretized diffusion equation
    nt: int
        number of time steps
    sigma: float
        alpha*td/dx^2

    qdx: float
        flux at right boundary * dx
    Returns:
    -------
    T: array of floats
        temperature profile after nt time steps
	    Example :
	--------------
	L = 1.
	nt = 100
	nx = 51
	alpha = 1.22e-3

	q = 0.
	dx = L/(nx-1)
	qdx = q*dx

	Ti = numpy.zeros(nx)
	Ti[0] = 100
	sigma = 0.5
	dt = sigma * dx*dx/alpha
	nt = 1000

	A = generateMatrix(nx, sigma)
	T = implicit_ftcs(Ti.copy(), A, nt, sigma, qdx)
    """

    for t in range(nt):
        Tn = T.copy()
        b = generateRHS(Tn, sigma, qdx)
        # Use numpy.linalg.solve
        T_interior = solve(A,b)
        T[1:-1] = T_interior
        # Enforce Neumann BC (Dirichlet is enforced automatically)
        T[-1] = T[-2] + qdx

    return T

########################## Crank-Nicolson method

def generateMatrix_CN(N, sigma):
    """ Computes the matrix for the diffusion equation with Crank-Nicolson
        Dirichlet condition at i=0, Neumann at i=-1

    Parameters:
    ----------
    N: int
        Number of discretization points
    sigma: float
        alpha*dt/dx^2

    Returns:
    -------
    A: 2D numpy array of float
        Matrix for diffusion equation
    """

    # Setup the diagonal
    d = 2*numpy.diag(numpy.ones(N-2)*(1+1./sigma))

    # Consider Neumann BC
    d[-1,-1] = 1+2./sigma

    # Setup upper diagonal
    ud = numpy.diag(numpy.ones(N-3)*-1, 1)

    # Setup lower diagonal
    ld = numpy.diag(numpy.ones(N-3)*-1, -1)

    A = d + ud + ld

    return A

def generateRHS_CN(T, sigma):
    """ Computes right-hand side of linear system for diffusion equation
        with backward Euler

    Parameters:
    ----------
    T: array of float
        Temperature at current time step
    sigma: float
        alpha*dt/dx^2

    Returns:
    -------
    b: array of float
        Right-hand side of diffusion equation with backward Euler
    """

    b = T[1:-1]*2*(1./sigma-1) + T[:-2] + T[2:]
    # Consider Dirichlet BC
    b[0] += T[0]

    return b

def CrankNicolson(T, A, nt, sigma):
    """ Advances diffusion equation in time with Crank-Nicolson

    Parameters:
    ----------
    T: array of float
        initial temperature profile
    A: 2D array of float
        Matrix with discretized diffusion equation
    nt: int
        number of time steps
    sigma: float
        alpha*td/dx^2

    Returns:
    -------
    T: array of floats
        temperature profile after nt time steps

	    Example :
	--------------
	L = 1
	nx = 21
	alpha = 1.22e-3

	dx = L/(nx-1)

	Ti = numpy.zeros(nx)
	Ti[0] = 100

	sigma = 0.5
	dt = sigma * dx*dx/alpha
	nt = 10

	A = generateMatrix(nx, sigma)
	T = CrankNicolson(Ti.copy(), A, nt, sigma)

    """

    for t in range(nt):
        Tn = T.copy()
        b = generateRHS(Tn, sigma)
        # Use numpy.linalg.solve
        T_interior = solve(A,b)
        T[1:-1] = T_interior
        # Enforce Neumann BC (Dirichlet is enforced automatically)
        T[-1] = T[-2]

    return T
