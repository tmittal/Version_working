#  new domain class
#  every process should exist in the context of a domain
from PyMagmaCh.A1_domain.axis import Axis
from PyMagmaCh.utils import constants as const

class _Domain(object):
    def __str__(self):
        return ("PymagmaCh Domain object with domain_type=" + self.domain_type + " and shape=" +
                str(self.shape) + " and name = " + self.name)
    def __init__(self, axes=None, **kwargs):
        self.name = 'None'
        self.domain_type = 'undefined'
        # self.axes should be a dictionary of axes
        # make it possible to give just a single axis:
        self.axes = self._make_axes_dict(axes)
        self.numdims = len(self.axes.keys())
        shape = []
        axcount = 0
        axindex = {}
        #  ordered list of axes
        axlist = list(self.axes)
        #for axType, ax in self.axes.iteritems():
        for axType in axlist:
            ax = self.axes[axType]
            shape.append(ax.num_points)
            #  can access axes as object attributes - Access using self.axes
            ## setattr(self, axType, ax)
            axindex[axType] = axcount
            axcount += 1
        self.axis_index = axindex
        self.axcount = axcount
        self.shape = tuple(shape)

    def _make_axes_dict(self, axes):
        if type(axes) is dict:
            axdict = axes
        elif type(axes) is Axis:
            ax = axes
            axdict = {ax.axis_type: ax}
        elif axes is None:
            axdict = {'empty': None}
        else:
            raise ValueError('axes needs to be Axis object or dictionary of Axis object')
        return axdict

def box_model_domain(num_points=2, **kwargs):
    '''Create a box model domain (a single abstract axis).'''
    ax = Axis(axis_type='abstract', num_points=num_points,note='Box Model')
    boxes = _Domain(axes=ax, **kwargs)
    boxes.domain_type = 'box'
    return boxes

def make_slabatm_axis(num_points=1,bounds=[1,1e3]):
    '''Convenience method to create a simple axis for a slab atmosphere/surface.'''
    depthax = Axis(axis_type='depth', num_points=num_points, bounds=bounds)
    return depthax

class SlabAtmosphere(_Domain):
    def __init__(self, axes=make_slabatm_axis(), **kwargs):
        super(SlabAtmosphere, self).__init__(axes=axes, **kwargs)
        self.domain_type = 'atm'


def z_column(num_depth=30,depth=None, **kwargs):
    '''Convenience method to create domains for a single depth grid for magma chambers (1D model),
    assume that the x-y extent of the chamber is much larger than the height axis ..

    num_depth is the number of depth levels (evenly spaced from surface to moho depth)
    Returns a list of 1 Domain objects (z column)

    Usage:
    z_clmn = z_column()
        or
    z_clmn = z_column(num_depth=2)
    print z_clmn

    Can also pass a depth array or depth level axis object
    '''
    if depth is None:
        depthax = Axis(axis_type='depth', num_points=num_depth)
    elif isinstance(depth, Axis):
        depthax = depth
    else:
        try:
            depthax = Axis(axis_type='depth', points=depth)
        except:
            raise ValueError('depth must be Axis object or depth array')
    z_clmn   = _Domain(axes=depthax, **kwargs)
    z_clmn.domain_type = 'z_column'
    return z_clmn

def z_column_atm(num_depth=30,depth=None, **kwargs):
    '''Convenience method to create domains for a single depth grid for magma chambers (1D model),
    assume that the x-y extent of the chamber is much larger than the height axis ..

    num_depth is the number of depth levels (evenly spaced from surface to moho depth)
    Returns a list of 2 Domain objects (z column, slab atmosphere)

    Usage:
    z_clmn, atm_slab = z_column()
        or
    z_clmn, atm_slab = z_column(num_depth=2)
    print z_clmn, atm_slab

    Can also pass a depth array or depth level axis object
    '''
    if depth is None:
        depthax = Axis(axis_type='depth', num_points=num_depth)
    elif isinstance(depth, Axis):
        depthax = depth
    else:
        try:
            depthax = Axis(axis_type='depth', points=depth)
        except:
            raise ValueError('depth must be Axis object or depth array')
    atmax = Axis(axis_type='depth',num_points=1, bounds=[1, 1e3]) # set a slab atm/surface model
    atm_slab = SlabAtmosphere(axes=atmax, **kwargs)
    z_clmn   = _Domain(axes=depthax, **kwargs)
    z_clmn.domain_type = 'z_column'
    return z_clmn, atm_slab

def z_radial_column_atm(num_depth=90, num_rad=30, depth=None,
                      radial_val=None, **kwargs):
    '''Convenience method to create domains for a single depth grid for
    magma chambers (2D model) + radial grid; assume that the chamber shape is axisymmteric

    num_depth is the number of depth levels (evenly spaced from surface to moho depth)
    num_rad is the number of radial levels (evenly spaced from 0 to region_extent)
    Returns a list of 2 Domain objects (z column, slab atmosphere)

    Usage:
    z_clmn, atm_slab = z_radial_column()
        or
    z_clmn, atm_slab = z_radial_column(num_depth=90, num_rad=30)
    print z_clmn, atm_slab

    Can also pass a depth array or depth level axis object
    '''
    if depth is None:
        depthax = Axis(axis_type='depth', num_points=num_depth)
    elif isinstance(depth, Axis):
        depthax = depth
    else:
        try:
            depthax = Axis(axis_type='depth', points=depth)
        except:
            raise ValueError('depth must be Axis object or depth array')
    if radial_val is None:
        radax = Axis(axis_type='x_val', num_points=num_rad)
    elif isinstance(radial_val, Axis):
        radax = radial_val
    else:
        try:
            radax = Axis(axis_type='x_val', points=radial_val)
        except:
            raise ValueError('radial_val must be Axis object or x_val array')
    atmax = Axis(axis_type='depth',num_points=1, bounds=[1, 1e3]) # set a slab atm/surface model
    atm_slab = SlabAtmosphere(axes={'x_val':radax, 'depth':atmax}, **kwargs)
    z_clmn   = _Domain(axes={'x_val':radax, 'depth':depthax}, **kwargs)
    z_clmn.domain_type = 'z_column'
    return z_clmn, atm_slab


def z_radial_column(num_depth=90, num_rad=30, depth=None,
                      radial_val=None, **kwargs):
    '''Convenience method to create domains for a single depth grid for
    magma chambers (2D model) + radial grid; assume that the chamber shape is axisymmteric

    num_depth is the number of depth levels (evenly spaced from surface to moho depth)
    num_rad is the number of radial levels (evenly spaced from 0 to region_extent)
    Returns a list of 2 Domain objects (z column)

    Usage:
    z_clmn = z_radial_column()
        or
    z_clmn = z_radial_column(num_depth=90, num_rad=30)
    print z_clmn

    Can also pass a depth array or depth level axis object
    '''
    if depth is None:
        depthax = Axis(axis_type='depth', num_points=num_depth)
    elif isinstance(depth, Axis):
        depthax = depth
    else:
        try:
            depthax = Axis(axis_type='depth', points=depth)
        except:
            raise ValueError('depth must be Axis object or depth array')
    if radial_val is None:
        radax = Axis(axis_type='x_val', num_points=num_rad)
    elif isinstance(radial_val, Axis):
        radax = radial_val
    else:
        try:
            radax = Axis(axis_type='x_val', points=radial_val)
        except:
            raise ValueError('radial_val must be Axis object or x_val array')
    z_clmn   = _Domain(axes={'x_val':radax, 'depth':depthax}, **kwargs)
    z_clmn.domain_type = 'z_column'
    return z_clmn
