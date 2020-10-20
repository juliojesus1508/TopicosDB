from os import path
import sys

sys.path.append("..")
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from values.documents import *
from values.stopwords import *

import string
import numpy as np

import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer