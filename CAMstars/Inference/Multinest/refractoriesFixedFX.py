'''
This script uses Nested Sampling to infer the refractory fraction of each element
from data of a population of stars.

The abundance of element X in star i is taken to be

X_i = X_t ((1-f_i) + f ((1 - f_X) + f_X delta_{d,*})),

where f_i are the photospheric fractions of the stars, f_X are the refractory fractions,
and delta_{d,*} are the enhancement/depletion fractions for the star. X_t are taken
to be fixed reference values.

In this inference problem we hold f_X = 1 for all X with condensation temperatures above 1500K.
'''

import numpy as np
import os
from CAMstars.Inference.Multinest.multinestWrapper import run, analyze, plot1D, plot2D
from CAMstars.Parsers.stars import accretingPop, AJMartinPop, LFossatiPop, sol
from CAMstars.Parsers.condensation import condenseTemps
from CAMstars.AccretedFraction.star import star
from CAMstars.Material.population import population
from CAMstars.Misc.constants import mSun, yr
from CAMstars.Misc.utils import propagate_errors, gaussianLogLike

# Combine the field populations
field = AJMartinPop + LFossatiPop

# Filter out stars with no known Mdot
accretingPop = population([m for m in accretingPop.materials if np.isfinite(m.params['logmdot'])])
stars = accretingPop.materials

# Extract accreted fractions
logf = np.array(list(s.params['logfAcc'] for s in stars))
dlogf = np.array(list(s.params['dlogfAcc'] for s in stars))

#elements = list(e for e in accretingPop.species if e in field.species)
elements = ['He','C','O','S','Ca','Sr','Fe','Mg','Si']#,'Al','Ti','Sc','Ni','Mn','Zn','V','Na']

# Sort elements by condensation temperature
elements = sorted(elements, key=lambda x: condenseTemps[x])

fixedElements = list(e for e in elements if condenseTemps[e] > 1500)
freeElements = list(e for e in elements if condenseTemps[e] <= 1500)

diff = list([star.logX[i] - field.queryStats(e)[0] for i,e in enumerate(star.names) if e in elements] for star in stars)
var = list([field.queryStats(e)[1]**2 + star.dlogX[i]**2 for i,e in enumerate(star.names) if e in elements] for star in stars)

# The formalism has trouble with fixing some parameters but not others, so we assign an error of 0.01 to any logf's that have zero error.
dlogf[dlogf == 0] += 0.01

def probability(params):
	nS = len(stars)
	logfAcc = params[:nS]
	logd = params[nS:2*nS]
	fX = params[2*nS:]

	# Expand fX to include fixed elements
	fX = np.array(list(fX[freeElements.index(e)] if e in freeElements else 1 for e in elements))

	fAcc = 10**logfAcc
	fAcc[fAcc > 1] = 1

	q = [[np.log((1-fAcc[i]) + fAcc[i] * (1-fX[elements.index(e)] + 10**(logd[i])*fX[elements.index(e)])) for e in m.names if e in elements] for i,m in enumerate(stars)]

	like = [[gaussianLogLike((diff[i][j] - q[i][j])/var[i][j]**0.5) for j in range(len(q[i]))] for (i,m) in enumerate(stars)]
	like = sum(sum(l) for l in like)
	like += np.sum(gaussianLogLike((logfAcc - logf) / dlogf))

	return like

dir_path = os.path.dirname(os.path.realpath(__file__))
oDir = dir_path + '/../../../Output/RefractoriesFixedFX/'
oDir = os.path.abspath(oDir)
oPref = 'Ref'
parameters = [s.name + ' $\log f$' for s in stars] + [s.name + ' $\log \delta$' for s in stars] + ['$f_{\mathrm{' + e + '}}$' for e in freeElements]
ranges = [(lf - 3 * dlf,min(0, lf + 3 * dlf)) for lf, dlf in zip(*(logf, dlogf))] + len(stars) * [(-3,3)] + len(freeElements) * [(0,1)]
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
# Expand fX to include fixed elements
fX = np.array(list(fX[freeElements.index(e)] if e in freeElements else 1 for e in elements))

fAcc = 10**np.array(logfAcc)
fAcc[fAcc > 1] = 1

solX = [[sol.query(e)[0] for e in m.names if e in elements] for i,m in enumerate(stars)]
solV = [[sol.query(e)[1] for e in m.names if e in elements] for i,m in enumerate(stars)]
refs = [[field.queryStats(e)[0] for e in m.names if e in elements] for i,m in enumerate(stars)]
refsv = [[field.queryStats(e)[1] for e in m.names if e in elements] for i,m in enumerate(stars)]
model = [[field.queryStats(e)[0] + np.log((1-fAcc[i]) + fAcc[i] * (1-fX[elements.index(e)] + 10**(logd[i])*fX[elements.index(e)])) for e in m.names if e in elements] for i,m in enumerate(stars)]
outs = [[m.query(e)[0] for e in m.names if e in elements] for m in stars]
outsv = [[m.query(e)[1] for e in m.names if e in elements] for m in stars]
outsn = [[e for e in m.names if e in elements] for m in stars]

solX = list(np.array(s) for s in solX)
solV = list(np.array(s) for s in solV)
refs = list(np.array(r) for r in refs)
refsv = list(np.array(r) for r in refsv)
model = list(np.array(m) for m in model)
outs = list(np.array(o) for o in outs)
outsv = list(np.array(o) for o in outsv)
outsn = list(np.array(o) for o in outsn)

for i,star in enumerate(stars):
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.errorbar(range(len(model[i])), solX[i], yerr=solV[i], c='c')
	ax.errorbar(range(len(model[i])), refs[i], yerr=refsv[i], c='k')
	ax.scatter(range(len(model[i])), model[i],c='b')
	ax.errorbar(range(len(model[i])),outs[i],yerr=outsv[i], fmt='o',c='r')
	ax.set(xticks=range(len(model[i])), xticklabels=outsn[i])
	ax.set_ylabel('$\log [X]$')
	ax = fig.add_subplot(212)
	ax.scatter(range(len(model[i])), outs[i] - model[i],c='b')
	ax.set(xticks=range(len(model[i])), xticklabels=outsn[i])
	ax.set_ylabel('Residuals')
	plt.savefig(oDir + '/' + star.name + '_model.pdf')
	plt.clf()


plot1D(a, parameters, oDir, oPref)
plot2D(a, parameters, oDir, oPref)







