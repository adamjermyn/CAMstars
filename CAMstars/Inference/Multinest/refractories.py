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
from CAMstars.Inference.Multinest.multinestWrapper import run, analyze, plot1D, plot2D
from CAMstars.Parsers.accretingStars import accretingPop
from CAMstars.Parsers.AJMartinStars import AJMartinPop
from CAMstars.Parsers.LFossatiStars import LFossatiPop
from CAMstars.AccretedFraction.star import star
from CAMstars.Misc.constants import mSun, yr
from CAMstars.Misc.utils import propagate_errors

# Combine the field populations
field = AJMartinPop + LFossatiPop

# Calculate accreted fractions
def fraction(mass, radius, temperature, u_rot, logmdot):

	# u_rot is vsini in km/s
	# logmdot is non-dimensionalised by Msun/yr
	# Now we convert these to C.G.S.

	mdot = 10**(logmdot) * (mSun / yr)
	u_rot *= 1e5

	s = star(mass, radius, temperature)

	# For this we assume that the accreted material is lighter than the stellar material
	return s.findF(u_rot, mdot, gradient=False) 

# Wrapper so the input is multidimensional
frac = lambda x: fraction(*x)

x = [(p.params['M'], p.params['R'], p.params['T'], p.params['vrot'], p.params['logmdot']) for p in accretingPop.materials]
dx = [(p.params['dM'], p.params['dR'], p.params['dT'], p.params['dvrot'], p.params['dlogmdot']) for p in accretingPop.materials]

f = [frac(y) for y in x]
df = [propagate_errors(frac, y, dy) for y,dy in zip(*(x, dx))]

f = np.array(f)
df = np.array(df)

print(f)
print(df)