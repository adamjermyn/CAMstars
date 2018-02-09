import numpy as np
from pymultinest.solve import solve
import h5py
import json
import sys
import os
from material import material, masker
from starPrior import starPrior
from star import star

# Our parameter space is composed of:
#	- Stellar temperature
#	- Stellar vsini
#	- Stellar sini
#	- Stellar mass
#	- Stellar radius
#	- Log(Accretion Rate)
#	- Bulk composition of the star
#	- Composition of the accreting material

def Prior(cube, ranges):
	for i in range(len(cube)):
		cube[i] = cube[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0] 
	return cube

def bulkPrior(solMat, bulk):
	# We take as our prior on the star's bulk composition the Sun's likelihood.
	# The log prior over abundances is then sol.likelihood(bulk).
	return solMat.likelihood(bulk)

def accretePrior(solMat, accrete, tCut, dT, delta):
	# For the accreted material we take the same prior, but for all non-H/non-He elements
	# we assume that they are depleted by an amount that depends on their condensation temperature.
	return solMat.likelihood(accrete - np.log10(solMat.depletedFactor(tCut, dT, delta)))

def globalPrior(params, solMat, starStar):
	# Global parameters go first, then abundances.
	sParams = params[:6]
	delta = params[6]
	tCut = params[7]
	dT = params[8]
	bulk = params[9:len(solMat.X) + 9]
	accrete = params[9 + len(solMat.X):2*len(solMat.X) + 9]
	return bulkPrior(solMat, bulk) + starStar.prior(sParams) + accretePrior(solMat, accrete, tCut, dT, delta)

def model(params, starMat):
	t = params[0]
	vs = params[1]
	i = params[2]
	m = params[3]
	r = params[4]
	mdot = params[5]
	s = star(m, radius=r, temperature=t)
	f = s.findF(vs / (i + 1e-5), mdot, gradient=False)
	bulk = params[9:len(starMat.X) + 9]
	accrete = params[9 + len(starMat.X):2*len(starMat.X) + 9]
	abundance = np.log10(10**bulk * (1-10**f) + 10**accrete * 10**f)
	return abundance

def globalLikelihood(params, starMat):
	m = model(params, starMat)
	return starMat.likelihood(m)

def probability(params, starMat, starStar, solMat):
	logp = globalLikelihood(params, starMat) + globalPrior(params, solMat, starStar)
#	return -np.sqrt(-logp)	
	return logp

def run(starName):
	solMat = material('Sol')
	starMat = material(starName)
	starStar = starPrior(starName)
	masker(solMat, starMat)

	dirname = '../Output/Multi/' + starName + '_star'

	if not os.path.exists(dirname):
		os.makedirs(dirname)

	prefix = dirname + '/'+starName+'_star_'
	ndim = len(solMat.X) + len(starMat.X) + 9
	ranges = starStar.ranges + [(-3,3),(0,2000),(-5,5)] + [(-2 + x, 2 + x) for x in solMat.X] + [(-2 + x, 2 + x) for x in solMat.X]

	def pri(x):
		return Prior(x, ranges)

	def prob(x):
		return probability(x, starMat, starStar, solMat)

	result = solve(LogLikelihood=prob, Prior=pri, n_dims = ndim, importance_nested_sampling=False,resume=False,outputfiles_basename=prefix,verbose=True, n_live_points=500)

	print()
	print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
	print()
	print('parameter values:')

	parameters = ["$T$", "$v\sin i$", "$\sin i$", "$M$", "$R$", "$\log \dot{M}$","$\log \delta_X$","$T_c (\mathrm{K})$","$(\delta T/\mathrm{K})^{-1} $"] + solMat.names + solMat.names
	n_params = len(parameters)

	for name, col in zip(parameters, result['samples'].transpose()):
		print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

	with open('%sparams.json' % prefix, 'w') as f:
		json.dump(parameters, f, indent=2)

starName = sys.argv[1]
run(starName)
