'''
This script uses Nested Sampling to infer the refractory fraction of each element
from data of a population of stars.

The abundance of element X in star i is taken to be

X_i = X_t ((1-f_i) + f ((1 - f_X) + f_X delta_{d,*})),

where f_i are the photospheric fractions of the stars, f_X are the refractory fractions,
and delta_{d,*} are the enhancement/depletion fractions for the star. X_t are taken
to be fixed reference values.
'''

import numpy as np
from multinestWrapper import run, analyze, plot1D, plot2D
from CAMstars.Parsers.accretingStars import accretingPop
from CAMstars.Parsers.AJMartinStars import AJMartinPop
from CAMstars.Parsers.LFossatiStars import LFossatiPop

# Combine the field populations
field = AJMartinPop + LFossatiPop

# Calculate accreted fractions
