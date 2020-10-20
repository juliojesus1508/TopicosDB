from os import path
import sys

#sys.path.append("..")
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import pandas as pd

from values.documents import *
from boolean_model.model import BooleanModel


