'''
This script uses Nested Sampling to infer the refractory fraction of each element
from data of a population of stars.

The abundance of element X in star i is taken to be

X_i = X_t ((1-f_i) + f ((1 - f_X) + f_X delta_{d,*})),

where f_i are the photospheric fractions of the stars, f_X are the refractory fractions,
and delta_{d,*} are the enhancement/depletion fractions for the star. X_t are taken
to be fixed reference values.

In this inference problem we hold f_X = 1 for all X with condensation temperatures above 1000K
and f_X = 0 for those with T_c < 200K.
'''

import numpy as np
import os
from CAMstars.Inference.Multinest.multinestWrapper import run, analyze, plot1D, plot2D
from CAMstars.Parsers.condensation import condenseTemps
from CAMstars.AccretedFraction.star import star
from CAMstars.Material.population import population
from CAMstars.Misc.constants import mSun, yr
from CAMstars.Misc.utils import propagate_errors, gaussianLogLike

from CAMstars.Parsers.stars import sol
from CAMstars.Parsers.whitelist_pop_filter import field, accretingPop, stars

# Extract accreted fractions
logf = np.zeros(len(stars))
dlogf = np.array(list(s.params['dlogfAcc'] for s in stars))

elements = list(e for e in accretingPop.species if e in field.species)

# Sort elements by condensation temperature
elements = sorted(elements, key=lambda x: condenseTemps[x])

fixedList1 = ['Fe','Mg','Si','Ti']
fixedList0 = ['H','He']
freeList = ['C','O','Zn','Na','S']

def indicator(x):
	if x in fixedList1:
		return 1
	elif x in fixedList0:
		return 0
	else:
		return None

fixedElements = {e:indicator(e) for e in elements}
freeElements = list(e for e in elements if fixedElements[e] is None)

diff = list([star.logX[i] - field.queryStats(e)[0] for i,e in enumerate(star.names) if e in elements] for star in stars)
var = list([field.queryStats(e)[1]**2 + star.dlogX[i]**2 for i,e in enumerate(star.names) if e in elements] for star in stars)

# The formalism has trouble with fixing some parameters but not others, so we assign an error of 0.01 to any logf's that have zero error.
dlogf[dlogf == 0] += 0.01

def probability(params):
	nS = len(stars)
	logd = params[:nS]
	fX = params[nS:2*nS]

	# Expand fX to include fixed elements
	fX = np.array(list(fX[freeElements.index(e)] if e in freeElements else fixedElements[e] for e in elements))

	fAcc = 10**logf

	q = [[np.log((1-fAcc[i]) + fAcc[i] * (1-fX[elements.index(e)] + 10**(logd[i])*fX[elements.index(e)])) for e in m.names if e in elements] for i,m in enumerate(stars)]

	like = [[gaussianLogLike((diff[i][j] - q[i][j])/var[i][j]**0.5) for j in range(len(q[i]))] for (i,m) in enumerate(stars)]
	like = sum(sum(l) for l in like)

	return like

dir_path = os.path.dirname(os.path.realpath(__file__))
oDir = dir_path + '/../../../../Output/FixedFX_Subset_fAcc1/'
oDir = os.path.abspath(oDir)
oPref = 'Ref'
parameters = [s.name + ' $\log \delta$' for s in stars] + ['$f_{\mathrm{' + e + '}}$' for e in freeElements]
ranges = len(stars) * [(-3,3)] + len(freeElements) * [(0,1)]
ndim = len(ranges)

for i,p in enumerate(parameters):
	print(i, p)

print(elements)
print(freeElements)

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
fX = np.array(list(fX[freeElements.index(e)] if e in freeElements else fixedElements[e] for e in elements))

fAcc = 10**np.array(logfAcc)

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

import pickle
pickle.dump(a, open(oDir+'/' + 'analyzer.pickle','wb'))


