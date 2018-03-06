import numpy as np


class Field(np.ndarray):
    '''Custom class for PyMagmaCh gridded quantities, called Field
    This class behaves exactly like numpy.ndarray
    but every object has an attribute called domain
    which is the domain associated with that field (e.g. state variables)
    as well as an axes in the domain that the field refers to.
    Inputs - can either give axis variable (for single axis) or a dict with the axis_type
    '''

    def __new__(cls, input_array, domain=None,axis=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        # This should ensure that shape is (1,) for scalar input
        obj = np.atleast_1d(input_array).view(cls)
        # add the new attribute to the created instance
        if type(domain) is str :
            obj.domain = domain
        elif (domain != None) :
            obj.domain = domain.name
        else :
            obj.domain ='None'
        if (axis == None) :
            obj.axis = 'None'
        elif type(axis) is dict:
            obj.axis = axis
        elif type(axis) is str :
            obj.axis = axis
        else :
            obj.axis = axis.axis_type
        # Finally, we must return the newly created object:
        obj.name = 'None'
        return obj

    def append_val(self,val):
        # append either a value or a set of values to a given state
        self = np.append(self,val)
        return self


    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(Field, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. Field():
        #    obj is None
        #    (we're in the middle of the Field.__new__
        #    constructor, and self.domain will be set when we return to
        #    Field.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(Field):
        #    obj is arr
        #    (type(obj) can be Field)
        # From new-from-template - e.g statearr[:3]
        #    type(obj) is Field
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'domain', because this
        # method sees all creation of default objects - with the
        # Field.__new__ constructor, but also with
        # arr.view(Field).
        self.domain = getattr(obj, 'domain', None)
        # We do not need to return anything

def global_mean(field):
    '''Calculate global mean of a field with depth dependence.'''
    try:
        dpth = field.domain.axes['depth'].points
    except:
        raise ValueError('No depth axis in input field.')
    arry = field.squeeze()
    delta_dpth = np.diff(dpth, n=1, axis=-1)
    delta_arry = (arry[1:] + arry[:-1])/2.
    # Assume that the surface is at z =0, +ve as one goes down.
    avg_val = np.sum(delta_arry*delta_dpth)/np.sum(delta_dpth)
    return avg_val
