import numpy as np
from pymultinest.solve import solve
import h5py
import json
import sys
import os
from material import material, masker

# Our parameter space is composed of:
#	- Bulk composition of the star
#	- Composition of the accreting material
#	- A few global parameters, namely accreted fraction, depletion factor, and possibly a few others.

def Prior(cube, ranges):
	for i in range(len(cube)):
		cube[i] = cube[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0] 
	return cube

def bulkPrior(sol, bulk):
	# We take as our prior on the star's bulk composition the Sun's likelihood.
	# The log prior over abundances is then sol.likelihood(bulk).
	return sol.likelihood(bulk)

def accretePrior(sol, accrete, tCut, dT, delta):
	# For the accreted material we take the same prior, but for all non-H/non-He elements
	# we assume that they are depleted by an amount that depends on their condensation temperature.
	return sol.likelihood(accrete - np.log10(sol.depletedFactor(tCut, dT, delta)))

def globalPrior(params, sol):
	# Global parameters go first, then abundances.
	f = params[0]
	delta = params[1]
	tCut = params[2]
	dT = params[3]
	bulk = params[4:len(sol.X) + 4]
	accrete = params[4 + len(sol.X):2*len(sol.X) + 4]
	return bulkPrior(sol, bulk) + accretePrior(sol, accrete, tCut, dT, delta)

def model(params, star):
	f = params[0]
	if f > 0:
		f = 0
	bulk = params[4:len(star.X) + 4]
	accrete = params[4 + len(star.X):2*len(star.X) + 4]
	abundance = np.log10(10**bulk * (1-10**f) + 10**accrete * 10**f)
	return abundance

def globalLikelihood(params, star):
	m = model(params, star)
	return star.likelihood(m)

def probability(params, star, sol):
	logp = globalLikelihood(params, star) + globalPrior(params, sol)
	return logp

def run(starName):
	sol = material('Sol')
	star = material(starName)
	masker(sol, star)

	dirname = '../Output/Multi/' + starName

	if not os.path.exists(dirname):
		os.makedirs(dirname)

	prefix = dirname + '/'+starName+'_'
	ndim = len(sol.X) + len(star.X) + 4	
	ranges = [(-8,0),(-3,3),(0,2000),(-5,5)] + [(-2 + x, 2 + x) for x in sol.X] + [(-2 + x, 2 + x) for x in sol.X]

	def pri(x):
		return Prior(x, ranges)

	def prob(x):
		return probability(x, star, sol)

	result = solve(LogLikelihood=prob, Prior=pri, n_dims = ndim, importance_nested_sampling=False,resume=False,outputfiles_basename=prefix,verbose=True, n_live_points=500)

	print()
	print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
	print()
	print('parameter values:')

	parameters = ["$\log f$","$\log \delta_X$","$T_c (\mathrm{K})$","$(\delta T/\mathrm{K})^{-1} $"] + sol.names + sol.names
	n_params = len(parameters)

	for name, col in zip(parameters, result['samples'].transpose()):
		print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

	with open('%sparams.json' % prefix, 'w') as f:
		json.dump(parameters, f, indent=2)


run(sys.argv[1])
