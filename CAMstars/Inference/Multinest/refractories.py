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
import os
from CAMstars.Inference.Multinest.multinestWrapper import run, analyze, plot1D, plot2D
from CAMstars.Parsers.accretingStars import accretingPop
from CAMstars.Parsers.AJMartinStars import AJMartinPop
from CAMstars.Parsers.LFossatiStars import LFossatiPop
from CAMstars.AccretedFraction.star import star
from CAMstars.Material.population import population
from CAMstars.Misc.constants import mSun, yr
from CAMstars.Misc.utils import propagate_errors, gaussianLogLike

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

# Wrapper so the input is multidimensional and the output is in log-space.
def frac(x):
	f0 = np.log10(fraction(*x))
	if f0 > 0:
		f0 = 0
	return f0

# Filter out stars with no known Mdot
accretingPop = population([m for m in accretingPop.materials if np.isfinite(m.params['logmdot'])])

x = [(p.params['M'], p.params['R'], p.params['T'], p.params['vrot'], p.params['logmdot']) for p in accretingPop.materials]

# We're using the symmetric version of the Mdot errors here.
dx = [(p.params['dM'], p.params['dR'], p.params['dT'], p.params['dvrot'], p.params['dlogmdot']) for p in accretingPop.materials]

logf = [frac(y) for y in x]
dlogf = [propagate_errors(frac, y, dy) for y,dy in zip(*(x, dx))]

logf = np.array(logf)
dlogf = np.array(dlogf)

elements = accretingPop.species
stars = accretingPop.materials

elements = ['He','C','O','S','Ca','Sr','Fe','Mg','Si']#,'Al','Ti','Sc','Ni','Mn','Zn','V','Na']

diff = list([star.logX[i] - field.queryStats(e)[0] for i,e in enumerate(star.names) if e in elements] for star in stars)
var = list([field.queryStats(e)[1]**2 + star.dlogX[i]**2 for i,e in enumerate(star.names) if e in elements] for star in stars)

# The formalism has trouble with fixing some parameters but not others, so we assign an error of 0.01 to any logf's that have zero error.
dlogf[dlogf == 0] += 0.01

def probability(params):
	nS = len(stars)
	logfAcc = params[:nS]
	logd = params[nS:2*nS]
	fX = params[2*nS:]

	fAcc = 10**logfAcc
	fAcc[fAcc > 1] = 1

	q = [[np.log((1-fAcc[i]) + fAcc[i] * (1-fX[elements.index(e)] + 10**(logd[i])*fX[elements.index(e)])) for e in m.names if e in elements] for i,m in enumerate(stars)]

	like = [[gaussianLogLike((diff[i][j] - q[i][j])/var[i][j]**0.5) for j in range(len(q[i]))] for (i,m) in enumerate(stars)]
	like = sum(sum(l) for l in like)
	like += np.sum(gaussianLogLike((logfAcc - logf) / dlogf))

	return like

dir_path = os.path.dirname(os.path.realpath(__file__))
oDir = dir_path + '/../../../Output/Refractories/'
oDir = os.path.abspath(oDir)
oPref = 'Ref'
parameters = [s.name + ' $\log f$' for s in stars] + [s.name + ' $\log \delta$' for s in stars] + ['$f_{\mathrm{' + e + '}}$' for e in elements]
ranges = [(lf - 3 * dlf,min(0, lf + 3 * dlf)) for lf, dlf in zip(*(logf, dlogf))] + len(stars) * [(-3,3)] + len(elements) * [(0,1)]
ndim = len(ranges)

for i,p in enumerate(parameters):
	print(i, p)

run(oDir, oPref, ranges, parameters, probability)
a, meds = analyze(oDir, oPref, oDir, oPref)

# Plot abundances
import matplotlib.pyplot as plt
plt.style.use('ggplot')

nS = len(stars)
logfAcc = meds[:nS]
logd = meds[nS:2*nS]
fX = meds[2*nS:]

fAcc = 10**np.array(logfAcc)
fAcc[fAcc > 1] = 1

model = [[field.queryStats(e)[0] + np.log((1-fAcc[i]) + fAcc[i] * (1-fX[elements.index(e)] + 10**(logd[i])*fX[elements.index(e)])) for e in m.names if e in elements] for i,m in enumerate(stars)]
outs = [[m.query(e)[0] for e in m.names if e in elements] for m in stars]
outsv = [[m.query(e)[1] for e in m.names if e in elements] for m in stars]
outsn = [[e for e in m.names if e in elements] for m in stars]

for i,star in enumerate(stars):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(range(len(model[i])), model[i],c='b')
	ax.errorbar(range(len(model[i])),outs[i],yerr=outsv[i], fmt='o',c='r')
	ax.set(xticks=range(len(model[i])), xticklabels=outsn[i])
	ax.set_ylabel('$\log [X]$')
	plt.savefig(oDir + '/' + star.name + '_model.pdf')
	plt.clf()

plot1D(a, parameters, oDir, oPref)
plot2D(a, parameters, oDir, oPref)







