import time, copy
import numpy as np
from PyMagmaCh.A1_domain.field import Field
from PyMagmaCh.A1_domain.domain import _Domain
from PyMagmaCh.utils import walk

#  New organizing principle:
#  processes are free to use object attributes to store whatever useful fields
#  the need. If later revisions require some special action to occur upon
#  getting or setting the attribute, we can always implement a property to
#  make that happen with no change in the API
#
#  the diagnostics dictionary will instead be used expressly for the purpose
#  of passing fields up the process tree!
#  so will often only be set by a parent process (and given a name that is
#  appropriate from the point of view of the parent process)

def _make_dict(arg, argtype):
    if arg is None:
        return {}
    elif type(arg) is dict:
        return arg
    elif isinstance(arg, argtype):
        return {arg.name: arg}
    else:
        raise ValueError('Problem with input type')


class Process(object):
    '''A generic parent class for all PyMagmaCh process objects.
    Every process object has a set of state variables on a spatial grid.
    Note for input - can either give a single domain or a dict with {domain.name:domain,....}
    Note - state can either be a single field or a dictionary of fields with {field_name :field,....}
         - subprocess can either be a single subprocess or a dictionary of subprocess
    '''
    def __str__(self):
        str1 = 'PyMagmaCh Process of type {0}. \n'.format(type(self))
        str1 += 'State variables and domain shapes: \n'
        for varname in self.state.keys():
            str1 += '  {0}: {1} \n'.format(varname, self.domains[varname].shape)
        str1 += 'The subprocess tree: \n'
        str1 += walk.process_tree(self)
        return str1

    def __init__(self, state=None, domains=None, subprocess=None,
                 diagnostics=None,name_inp=None ,**kwargs):
        # dictionary of domains. Keys are the domain names
        self.domains = _make_dict(domains, _Domain)
        # dictionary of state variables (all of type Field)
        self.domains_var = {}
        self.state = {}
        states = _make_dict(state, Field)
        if (state != None):
            for name, value in states.items():
                self.set_state(name, value)
        # dictionary of model parameters
        self.param = kwargs
        # dictionary of diagnostic quantities
        self.diagnostics = _make_dict(diagnostics, Field)
        self.creation_date = time.strftime("%a, %d %b %Y %H:%M:%S %z",
                                           time.localtime())
        if (name_inp != None) :
            self.name = name_inp
        else :
            self.name = 'None'
        # subprocess is either a single sub-processes or a dictionary of any sub-processes
        if subprocess is None:
            self.subprocess = {}
        else:
            self.add_subprocesses(subprocess)

    def add_subprocesses(self, procdict):
        '''Add a dictionary of subproceses to this process.
        procdict is dictionary with process names as keys.

        Can also pass a single process, which will be called \'default\'
        '''
        if isinstance(procdict, Process):
            self.add_subprocess(procdict.process_name, procdict)
        else:
            for name, proc in procdict.items():
                self.add_subprocess(name, proc)

    def add_subprocess(self, name, proc):
        '''Add a single subprocess to this process.
        name: name of the subprocess (str)
        proc: a Process object.'''
        if isinstance(proc, Process):
            self.subprocess.update({name: proc})
            self.has_process_type_list = False
        else:
            raise ValueError('subprocess must be Process object')

    def remove_subprocess(self, name):
        '''Remove a single subprocess from this process.
        name: name of the subprocess (str)'''
        self.subprocess.pop(name, None)
        self.has_process_type_list = False

    def set_state(self, name, value):
        '''Can either be for the first time - value is a field or
        subsequently changing the value of the field by passing a value array and field_name'''
        if isinstance(value, Field):
            # populate domains dictionary with domains from state variables
            self.domains_var.update({name: value.domain})
        else:
            try:
                thisdom = self.state[name].domain
                thisaxis = self.state[name].axis
            except:
                raise ValueError('State variable needs a domain.')
            value = np.atleast_1d(value)
            value = Field(value, domain=thisdom,axis=thisaxis)
        # set the state dictionary
        self.state[name] = value
        #setattr(self, name, value)

    # Some handy shortcuts... only really make sense when there is only
    # a single axis of that type in the process.
    @property
    def x_val(self):
        try:
            for domname, dom in self.domains.items():
                try:
                    thisxval = dom.axes['x_val'].points
                except:
                    pass
            return thisxval
        except:
            raise ValueError('Can\'t resolve an x_val axis - No domains.')

    @property
    def x_val_bounds(self):
        try:
            for domname, dom in self.domains.items():
                try:
                    thisxval = dom.axes['x_val'].bounds
                except:
                    pass
            return thisxval
        except:
            raise ValueError('Can\'t resolve an x_val axis  - No domains.')
    @property
    def y_val(self):
        try:
            for domname, dom in self.domains.items():
                try:
                    thisyval = dom.axes['y_val'].points
                except:
                    pass
            return thisyval
        except:
            raise ValueError('Can\'t resolve a y_val axis  - No domains.')
    @property
    def y_val_bounds(self):
        try:
            for domname, dom in self.domains.items():
                try:
                    thisyval = dom.axes['y_val'].bounds
                except:
                    pass
            return thisyval
        except:
            raise ValueError('Can\'t resolve a y_val axis  - No domains.')
    @property
    def depth(self):
        try:
            for domname, dom in self.domains.items():
                try:
                    thisdepth = dom.axes['depth'].points
                except:
                    pass
            return thisdepth
        except:
            raise ValueError('Can\'t resolve a depth axis  - No domains.')
    @property
    def depth_bounds(self):
        try:
            for domname, dom in self.domains.items():
                try:
                    thisdepth = dom.axes['depth'].bounds
                except:
                    pass
            return thisdepth
        except:
            raise ValueError('Can\'t resolve a depth axis  - No domains.')


def process_like(proc):
    '''Return a new process identical to the given process.
    The creation date is updated.'''
    newproc = copy.deepcopy(proc)
    newproc.creation_date = time.strftime("%a, %d %b %Y %H:%M:%S %z",
                                          time.localtime())
    return newproc


def get_axes(process_or_domain):
    '''Return a dictionary of all axes in Process or domain or dict of domains.'''
    if isinstance(process_or_domain, Process):
        dom = process_or_domain.domains
    else:
        dom = process_or_domain
    if isinstance(dom, _Domain):
        return dom.axes
    elif isinstance(dom, dict):
        axes = {}
        for thisdom in dom.values():
            assert isinstance(thisdom, _Domain)
            axes.update(thisdom.axes)
        return axes
    else:
        raise TypeError('dom must be a Process or domain or dictionary of domains.')
