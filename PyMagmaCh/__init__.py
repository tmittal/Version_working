__version__ = '0.1'

#  This list defines all the modules a user can load directly i.e.
#  as - from PyMagmaCh import xyz , instead of
#  from PyMagmaCh.utils import xyz ...

#  Note that the following command will load all the packages in this file
#  from PyMagmaCh import *
######################################################
######################################################
from PyMagmaCh.utils import constants


######################################################
#ok files :
#utils.constants
#utils.walk

######################################################

#from PyMagmaCh import radiation
#from PyMagmaCh.model.column import GreyRadiationModel, RadiativeConvectiveModel, BandRCModel
#from PyMagmaCh.model.ebm import EBM, EBM_annual, EBM_seasonal


# this should ensure that we can still import constants.py as PyMagmaCh.constants
#from PyMagmaCh.utils import thermo
# some more useful shorcuts
#from PyMagmaCh.model import ebm, column
#from PyMagmaCh.domain import domain
#from PyMagmaCh.domain.field import Field, global_mean
#from PyMagmaCh.domain.axis import Axis
#from PyMagmaCh.process.process import Process, process_like, get_axes
#from PyMagmaCh.process.time_dependent_process import TimeDependentProcess
#from PyMagmaCh.process.implicit import ImplicitProcess
#from PyMagmaCh.process.diagnostic import DiagnosticProcess
#from PyMagmaCh.process.energy_budget import EnergyBudget
#from PyMagmaCh.radiation.AplusBT import AplusBT
#from PyMagmaCh.radiation.AplusBT import AplusBT_CO2
#from PyMagmaCh.radiation.Boltzmann import Boltzmann
#from PyMagmaCh.radiation.insolation import FixedInsolation, P2Insolation, AnnualMeanInsolation, DailyInsolation
#from PyMagmaCh.radiation.radiation import Radiation
#from three_band import ThreeBandSW
#from PyMagmaCh.radiation.nband import NbandRadiation, ThreeBandSW
#from PyMagmaCh.radiation.water_vapor import ManabeWaterVapor
#from PyMagmaCh.dynamics.budyko_transport import BudykoTransport
