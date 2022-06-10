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
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
from CAMstars.Inference.Multinest.multinestWrapper import run, analyze, plot1D, plot2D
from CAMstars.Parsers.stars import accretingPop, AJMartinPop, LFossatiPop, sol
from CAMstars.Parsers.condensation import condenseTemps
from CAMstars.AccretedFraction.star import star
from CAMstars.Material.population import population
from CAMstars.Material.material import material
from CAMstars.Misc.constants import mSun, yr
from CAMstars.Misc.utils import propagate_errors, gaussianLogLike
import sys

# Load reference data

# Combine the field populations
field = AJMartinPop + LFossatiPop

# Exclude elements that have unreliable error estimates

exclude_Temp = [
'UCAC12284506',
'UCAC12284746',
'UCAC12065075',
'UCAC12284653',
'UCAC12284594',
'UCAC12065058',
'V380 Ori B'
]

for m in field.materials:
	if m.name in exclude_Temp:
		field.materials.remove(m)
for m in accretingPop.materials:
	if m.name in exclude_Temp:
		accretingPop.materials.remove(m)

exclude_S = [
'HD100546', 
'HD31648',
'HD36112',
'HD68695',
'HD179218',
'HD244604',
'HD123269',
'UCAC11105213',
'UCAC11105379',
'T Ori' 
]

for m in field.materials:
	if m.name in exclude_S:
		ind = m.queryIndex('S')
		if ind is not None:			
			m.names.pop(ind)
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

for m in accretingPop.materials:
	if m.name in exclude_S:
		ind = m.queryIndex('S')
		if ind is not None:
			m.names.pop(ind)
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

exclude_Zn = [
'UCAC11105106',
'UCAC11105213',
'UCAC11105379'
]

for m in accretingPop.materials:
	if m.name != 'HD144432':
		if 'Zn' in m.names:
			ind = m.queryIndex('Zn')
			if ind is not None:
				m.names.pop(ind)
				np.delete(m.logX, ind)
				np.delete(m.dlogX, ind)

for m in field.materials:
	if m.name in exclude_Zn:
		ind = m.queryIndex('Zn')
		if ind is not None:		
			m.names.pop(ind)
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

include_Na = [
'HD139614',
'HD144432'
]

for m in accretingPop.materials:
	if m.name not in include_Na:
		ind = m.queryIndex('Na')
		if ind is not None:
			m.names.pop(ind)
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

field = population(field.materials)
reference = field
accretingPop = population(accretingPop.materials)

def fElements(e):
	# Returns the refractory fraction for the element

	# Use S and Na from Sulfur paper. Oxygen is user-specified.
	# Others are just by condensation temperature.
	print(e)
	if e == 'S':
		return 0.866
	elif e == 'Na':
		return 0.69
	elif condenseTemps[e] > 1000:
		return 1
	else:
		return 0

def infer(s):
	# Performs an inference for one star.


	# Extract accreted fractions
	logf = s.params['logfAcc']
	dlogf = s.params['dlogfAcc']

	# The formalism has trouble with fixing some parameters but not others, so we assign an error of 0.01 to any logf's that have zero error.
	if dlogf == 0:
		dlogf += 0.01

	# Determine elements with enough information for inference
	elements = list(e for e in s.names if e in field.species and e in ['Fe'])

	# Sort elements by condensation temperature
	elements = sorted(elements, key=lambda x: condenseTemps[x])

	# Compute refractory fraction
	fX = list(fElements(e) for e in elements)

	# Compute difference and uncertainty
	diff = list(s.query(e)[0] - field.queryStats(e)[0] for e in elements)
	var = list(s.query(e)[1]**2 + field.queryStats(e)[1]**2 for e in elements)

	def probability(params):
		nS = len(stars)
		logfAcc = params[0]
		logd = params[1]

		fAcc = 10**logfAcc
		if fAcc > 1:
			fAcc = 1

		q = [np.log((1-fAcc) + fAcc * (1-fX[elements.index(e)] + 10**(logd)*fX[elements.index(e)])) for e in s.names if e in elements]

		like = [gaussianLogLike((diff[j] - q[j])/var[j]**0.5) for j in range(len(q))]
		like = sum(like)
		like += gaussianLogLike((logfAcc - logf) / dlogf)

		return like

	oDir = '/Users/ajermyn/out/GasDust_' + s.name + '/'
	oDir = os.path.abspath(oDir)
	oPref = 'Ref'
	parameters = [r'$\log f$',r'$\log delta$']
	ranges = [(logf - 3 * dlogf, min(0, logf + 3*dlogf)), (-3,3)]
	ndim = len(ranges)

	for i,p in enumerate(parameters):
		print(i, p)

	run(oDir, oPref, ranges, parameters, probability)
	a, meds = analyze(oDir, oPref, oDir, oPref)

	# Plot abundances
	import matplotlib.pyplot as plt
	plt.style.use('ggplot')

	nS = len(stars)
	logfAcc = meds[0]
	logd = meds[1]

	fAcc = 10**np.array(logfAcc)

	refX = [field.queryStats(e)[0] for e in elements]
	refV = [field.queryStats(e)[1] for e in elements]
	model = [field.queryStats(e)[0] + np.log((1-fAcc) + fAcc * (1-fX[elements.index(e)] + 10**(logd)*fX[elements.index(e)])) for e in elements]
	outX = [s.query(e)[0] for e in elements]
	outV = [s.query(e)[1] for e in elements]

	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.errorbar(range(len(model)), refX, yerr=refV, c='c')
	ax.scatter(range(len(model)), model,c='b')
	ax.errorbar(range(len(model)),outX,yerr=outV, fmt='o',c='r')
	ax.set(xticks=range(len(model)), xticklabels=elements)
	ax.set_ylabel('$\log [X]$')
	ax = fig.add_subplot(212)
	ax.scatter(range(len(model)), np.array(outX) - np.array(model),c='b')
	ax.set(xticks=range(len(model)), xticklabels=elements)
	ax.set_ylabel('Residuals')
	plt.savefig(oDir + '/model.pdf')
	plt.clf()

	plot1D(a, parameters, oDir, oPref)
	plot2D(a, parameters, oDir, oPref)

	import pickle
	pickle.dump(a, open(oDir+'/' + 'analyzer.pickle','wb'))

# Filter out stars with no known Mdot
accretingPop = population([m for m in accretingPop.materials if 'logfAcc' in m.params.keys() and 'dlogfAcc' in m.params.keys()])
stars = accretingPop.materials
for s in stars:
	infer(s)


