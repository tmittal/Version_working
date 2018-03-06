import numpy as np
import copy
from PyMagmaCh.utils import constants as const
from PyMagmaCh.process.process import Process
from PyMagmaCh.utils.walk import walk_processes

class TimeDependentProcess(Process):
    '''A generic parent class for all time-dependent processes.'''
    def __init__(self, time_type='explicit', timestep=None, topdown=True, **kwargs):
        # Create the state dataset
        super(TimeDependentProcess, self).__init__(**kwargs)
        self.tendencies = {}
        self.timeave = {}
        if timestep is None:
            self.set_timestep() # default is 1 yr ..
        else:
            self.set_timestep(timestep=timestep)
        self.time_type = time_type
        self.topdown = topdown
        self.has_process_type_list = False

    def compute(self):
        '''By default, the tendency (i.e d/dt (state) is zero  - no time change.
           Update state variables using all explicit tendencies
           Tendencies are d/dt(state) -- so just multiply by timestep for forward time
        '''
        for varname in self.state.keys():
            self.tendencies[varname] = np.zeros_like(self.state[varname])
            #for proc in self.process_types['explicit']:
            #    for varname in proc.state.keys():
            try: proc.state[varname] += (proc.tendencies[varname]*self.param['timestep'])
            except: pass


    def step_forward(self,no_update_time=True):
        '''new oop PyMagmaCh... just loop through processes
        and add up the tendencies'''
        if not self.has_process_type_list:
            self._build_process_type_list()
        # First compute all strictly diagnostic processes
        for proc in self.process_types['diagnostic']:
            proc.compute() # is typically defined in another sub-class
        # Compute diagnostics for all processes
        for proc in self.process_types['explicit']:
            proc.compute() # is typically defined in the another sub-class
        # Now compute all implicit processes -- matrix inversions
        for proc in self.process_types['implicit']:
            proc.compute()
            for varname in proc.state.keys():
                try: proc.state[varname] += proc.adjustment[varname]
                except: pass
        # Adjustment processes change the state instantaneously
        for proc in self.process_types['adjustment']:
            proc.compute()
            for varname, value in proc.state.items():
                #proc.set_state(varname, proc.adjusted_state[varname])
                try: proc.state[varname] += proc.adjustment[varname]
                except: pass
        # Gather all diagnostics
        for name, proc, level in walk_processes(self):
            self.diagnostics.update(proc.diagnostics)
            if (no_update_time == True):
                proc._update_time()

    def compute_diagnostics(self, num_iter=3):
        '''Compute all diagnostics, but don't update model state.
        By default it will call step_forward() 3 times to make sure all
        subprocess coupling is accounted for. The number of iterations can
        be changed with the input argument.
        The output is an updated process with states
        being the original but diagnostics updated'''
        this_state = copy.deepcopy(self.state)
        for n in range(num_iter):
            self.step_forward(no_update_time=False)
            for name, value in self.state.items():
                self.state[name][:] = this_state[name][:]

    def integrate_years(self, years=1.0, verbose=True):
        '''Timestep the model forward a specified number of years.'''
        days = years * const.days_per_year
        numsteps = int(self.time['num_steps_per_year'] * years)
        if verbose:
            print("Integrating for " + str(numsteps) + " steps, "
                  + str(days) + " days, or " + str(years) + " years.")
        #  begin time loop
        for count in range(numsteps):
            # Compute the timestep
            self.step_forward()
            if count == 0:
                # on first step only...
                # This implements a generic time-averaging feature
                # using the list of model state variables
                self.timeave = self.state.copy()
                # add any new diagnostics to the timeave dictionary
                self.timeave.update(self.diagnostics)
                for varname, value in self.timeave.items():
                    self.timeave[varname] = np.zeros_like(value)
            for varname in self.timeave.keys():
                try:
                    self.timeave[varname] += self.state[varname]
                except:
                    try:
                        self.timeave[varname] += self.diagnostics[varname]
                    except: pass
        for varname in self.timeave.keys():
            self.timeave[varname] /= numsteps  # this is simple averaged quantity of all the state fields + diagnostics ..
        if verbose:
            print("Total elapsed time is %s years."
                  % str(self.time['days_elapsed']/const.days_per_year))

    def integrate_converge(self, crit=1e-4, verbose=True):
        '''integrate until solution is converging (subsequently converge each state variable)
        param:  crit    - exit criteria for difference of iterated solutions
        '''
        for varname, value in self.state.items():
            value_old = copy.deepcopy(value)
            self.integrate_years(1,verbose=False)
            while np.max(np.abs(value_old-value)) > crit :
                value_old = copy.deepcopy(value)
                self.integrate_years(1,verbose=False)
        if verbose == True:
            print("Total elapsed time is %s years."
                  % str(self.time['days_elapsed']/const.days_per_year))

    def integrate_days(self, days=1.0, verbose=True):
        '''Timestep the model forward a specified number of days.'''
        years = days / const.days_per_year
        self.integrate_years(years=years, verbose=verbose)

    def _update_time(self):
        '''Increment the timestep counter by one.
        This function is called by the timestepping routines.'''
        self.time['steps'] += 1
        # time in days since beginning
        self.time['days_elapsed'] += self.time['timestep'] / const.seconds_per_day
        if self.time['day_of_year_index'] >= self.time['num_steps_per_year']-1:
            self._do_new_calendar_year()
        else:
            self.time['day_of_year_index'] += 1

    def _do_new_calendar_year(self):
        '''This function is called once at the end of every calendar year.'''
        self.time['day_of_year_index'] = 0  # back to Jan. 1
        self.time['years_elapsed'] += 1

    def set_timestep(self, timestep=const.seconds_per_year, num_steps_per_year=None):
        '''Change the timestep.
        Input is either timestep in seconds,
        or
        num_steps_per_year: a number of steps per calendar year (can be fractional <1).'''
        if num_steps_per_year is not None:
            timestep = const.seconds_per_year / num_steps_per_year
        else:
            num_steps_per_year = const.seconds_per_year / timestep
        timestep_days = timestep / const.seconds_per_day
        days_of_year = np.arange(0., const.days_per_year, timestep_days)
        self.time = {'timestep': timestep,
                     'num_steps_per_year': num_steps_per_year,
                     'day_of_year_index': 0,
                     'steps': 0,
                     'days_elapsed': 0,
                     'years_elapsed': 0,
                     'days_of_year': days_of_year}
        self.param['timestep'] = timestep

    def _build_process_type_list(self):
        '''Generate lists of processes organized by process type
        Currently, this can be 'diagnostic', 'explicit', 'implicit', or 'adjustment'.'''
        self.process_types = {'diagnostic': [], 'explicit': [], 'implicit': [], 'adjustment': []}
        for name, proc, level in walk_processes(self, topdown=self.topdown):
            self.process_types[proc.time_type].append(proc)
        self.has_process_type_list = True
