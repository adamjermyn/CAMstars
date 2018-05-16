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
from mpi4py import MPI
from CAMstars.Inference.Multinest.multinestWrapper import run, analyze, plot1D, plot2D
from CAMstars.Parsers.stars import accretingPop, AJMartinPop, LFossatiPop, sol
from CAMstars.Parsers.condensation import condenseTemps
from CAMstars.AccretedFraction.star import star
from CAMstars.Material.population import population
from CAMstars.Material.material import material
from CAMstars.Misc.constants import mSun, yr
from CAMstars.Misc.utils import propagate_errors, gaussianLogLike

# Load reference data
hdpref = '/home/asj42/Dropbox/Software/CAMstars/CAMstars/Inference/Multinest/GasDustRatio/'
fi = h5py.File(hdpref + 'table_gp.hdf','r')
output = np.array(fi['chemistry'])
output_d = np.array(fi['d_chemistry'])

print(np.min(output))

teff = np.array(fi['gridT'])
logg = np.array(fi['gridG'])
vsini = np.array(fi['gridV'])

# Load reference elements
fi = open(hdpref + 'elems.txt','r')
line = fi.readline()
line = line.split(',')
refElems = line[:-1]
for i in range(len(refElems)):
	# Make reference element names into standard format
	if len(refElems[i]) == 1:
		refElems[i] = refElems[i].upper()
	else:
		refElems[i] = refElems[i][0].upper() + refElems[i][1]

# Construct interpolator
reg = rgi((teff, logg, vsini), output, method='nearest', bounds_error=False, fill_value=None)
dreg = rgi((teff, logg, vsini), output_d, method='nearest', bounds_error=False, fill_value=None)


def buildReference(t, g, v):
	# Constructs a reference given a star
	abundances = reg((t,g,v))
	dabundances = dreg((t,g,v))

	for i,e in enumerate(refElems):
		dabundances[i] = (dabundances[i]**2 + sol.query(e)[1]**2)**0.5

	mat = material('Reference',refElems,abundances,dabundances)
	return mat

def fElements(e):
	# Returns the refractory fraction for the element

	# Use S and Na from Sulfur paper. Others are just by condensation temperature.
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

	# Calculate baseline
	teff = s.params['T']
	logg = s.params['logg']
	vsini = s.params['vrot']

	reference = buildReference(teff, logg, vsini)

	# Extract accreted fractions
	logf = s.params['logfAcc']
	dlogf = s.params['dlogfAcc']

	# The formalism has trouble with fixing some parameters but not others, so we assign an error of 0.01 to any logf's that have zero error.
	if dlogf == 0:
		dlogf += 0.01

	# Determine elements with enough information for inference
	elements = list(e for e in s.names if e in reference.names)

	# Sort elements by condensation temperature
	elements = sorted(elements, key=lambda x: condenseTemps[x])

	# Compute refractory fraction
	fX = list(fElements(e) for e in elements)

	# Compute difference and uncertainty
	diff = list(s.query(e)[0] - reference.query(e)[0] for e in elements)
	var = list(s.query(e)[1]**2 + reference.query(e)[1]**2 for e in elements)

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

	dir_path = os.path.dirname(os.path.realpath(__file__))
	oDir = dir_path + '/../../../../Output/GasDust_' + s.name + '/'
	oDir = os.path.abspath(oDir)
	oPref = 'Ref'
	parameters = ['$\log f$','$\log \delta$']
	ranges = [(logf - 3 * dlogf, min(0, logf + 3*dlogf)), (-3,3)]
	ndim = len(ranges)

	for i,p in enumerate(parameters):
		print(i, p)

	run(oDir, oPref, ranges, parameters, probability)

	if MPI.COMM_WORLD.Get_rank() == 0:
		a, meds = analyze(oDir, oPref, oDir, oPref)

		# Plot abundances
		import matplotlib.pyplot as plt
		plt.style.use('ggplot')

		nS = len(stars)
		logfAcc = meds[0]
		logd = meds[1]

		fAcc = 10**np.array(logfAcc)
		if fAcc > 1:
			fAcc = 1

		refX = np.array([reference.query(e)[0] - sol.query(e)[0] for e in elements])
		refV = np.array([reference.query(e)[1] for e in elements])
		model = np.array([reference.query(e)[0] - sol.query(e)[0] + np.log(1-fAcc + fAcc * (1-fX[elements.index(e)] + 10**(logd)*fX[elements.index(e)])) for e in elements])
		outX = np.array([s.query(e)[0] - sol.query(e)[0] for e in elements])
		outV = np.array([s.query(e)[1] for e in elements])

		print(refV)

		fig = plt.figure()
		ax = fig.add_subplot(211)
		ax.errorbar(range(len(model)), refX, yerr=refV, fmt='o',c='c',label='Reference')
		ax.scatter(range(len(model)), model,c='b',label='Model')
		ax.errorbar(range(len(model)),outX,yerr=outV, fmt='o',c='r',label='Observed')
		ax.set(xticks=range(len(model)), xticklabels=elements)
		ax.set_ylabel('$\log [X]/[X]_\odot$')
		ax.legend()
		ax = fig.add_subplot(212)
		ax.scatter(range(len(model)), outX - model,c='b',label='Residuals')
		ax.set(xticks=range(len(model)), xticklabels=elements)
		ax.set_ylabel('Residuals')
		ax.legend()
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


