import numpy as np
from PyMagmaCh.utils import constants as const


axis_types = ['depth', 'x_val', 'y_val', 'abstract']

# Implementing a simple cartesian distance axis type
# and probaly also an abstract dimensionless axis type (for box models) i.e. 0D models
# Note that the 0-D model can be thought of as something where we care only about when magma chmaber can erupt.

# Other axis types : depth (for 1D models), x_val and y_val (for a 2D and 3D model respectively)
#                    - can think of using only x_val in case of axisymmteric models (i.e. x_val = r_val).

# Note that bounds means an array with min and max value - output else would be the bounds used to gen each point (with point being the midpoint)
# Also, the saved bounds are alwats the bounds used to gen each point (with point being the midpoint)

class Axis(object):
    '''Create a new PyMagmaCh Axis object
    Valid axis types are:
        'depth'
        'x_val'
        'y_val'
        'abstract' (default)
    '''
    def __str__(self):
        return ("Axis of type " + self.axis_type + " with " +
                str(self.num_points) + " points.")

    def __init__(self, axis_type='abstract', num_points=10, points=None, bounds=None,note=None):
        if axis_type in axis_types:
            pass
        elif axis_type in ['y_value', 'Y_value']:
            axis_type = 'y_val'
        elif axis_type in ['x_value', 'X_value', 'R_value','R_val','r_val']:
            axis_type = 'x_val'
        elif axis_type in ['depth', 'Depth', 'chamberDepth', 'chamber_depth', 'zlab']:
            axis_type = 'depth'
        else:
            raise ValueError('axis_type %s not recognized' % axis_type)
        self.axis_type = axis_type

        defaultEndPoints = {'depth': (0., const.moho_depth),
                            'x_val': (0., const.region_size),
                            'y_val': (0., const.region_size),
                            'abstract': (0, num_points)}

        defaultUnits = {'depth': 'meters',
                        'x_val': 'meters',
                        'y_val': 'meters',
                        'abstract': 'none'}
        # if points and/or bounds are supplied, make sure they are increasing
        if points is not None:
            try:
                # using np.atleast_1d() ensures that we can use a single point
                points = np.sort(np.atleast_1d(np.array(points, dtype=float)))
            except:
                raise ValueError('points must be array_like.')
        if bounds is not None:
            try:
                bounds = np.sort(np.atleast_1d(np.array(bounds, dtype=float)))
            except:
                raise ValueError('bounds must be array_like.')

        if bounds is None:
            # assume default end points
            end0 = defaultEndPoints[axis_type][0]
            end1 = defaultEndPoints[axis_type][1]
            if points is not None:
                # only points are given - so use the default bounds in addition to the points.
                num_points = points.size
                df_set = np.diff(points)/2.
                bounds = points[:-1] + df_set
                bounds = np.insert(bounds,0, points[0]-df_set[0])
            else:
                # no points or bounds
                # create an evenly spaced axis
                delta = (end1 - end0) / num_points
                bounds = np.linspace(end0, end1, num_points+1)
                points = np.linspace(end0 + delta/2., end1-delta/2., num_points)
        else:  # bounds are given
            end0 = np.min(bounds)
            end1 = np.max(bounds)
            if points is None:
                # create an evenly spaced axis
                delta = (end1 - end0) / num_points
                bounds = np.linspace(end0, end1, num_points+1)
                points = np.linspace(end0 + delta/2., end1-delta/2., num_points)
            else:
                # points and bounds both given, check that they are compatible
                num_points = points.shape[0]
                bounds = np.linspace(end0, end1, num_points+1)
                if np.min(points) !=  end0:
                    raise ValueError('points and bounds are incompatible')
                if np.max(points) !=  end1:
                    raise ValueError('points and bounds are incompatible')
        self.note = note
        self.num_points = num_points
        self.units = defaultUnits[axis_type]
        self.points = points
        self.bounds = bounds
        self.delta = np.abs(np.diff(self.points))
