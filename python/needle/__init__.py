from . import ops
from .ops import *
from .autograd import Tensor
from .backend_numpy import cpu, all_devices

from . import init
from .init import ones, zeros, zeros_like, ones_like

from . import data
from . import nn
from . import optim
