import numpy as np
from pymultinest.solve import solve
import h5py
import json
import sys
import os
from glob import glob
from material import material

# Collect data files
files = glob('../Data/*.csv')
files = [f for f in files if 'condense' not in f and 'star' not in f and 'Sol' not in f]
files = files[:5]

# Identify relevant elements
materials = [material(f) for f in files]
elements = list(set([n for m in materials for n in m.names ]))

# Construct prior range
# Our parameter space is composed of:
#	- Bulk composition of stars
#	- For each star: accreted fraction, depletion factor, critical temperature
#
# Parameters are given in the order listed above.

# Identify sensible ranges for the prior.
# For each element we place the boundaries such that all stars are more than three sigma from the boundary,
# and we use the smallest window such that this is true.
ranges = {n:(np.inf, -np.inf) for n in elements}

for n in elements:
	for m in materials:
		ret = m.query(n)
		if ret is not None:
			x, dx = ret
			if x - 3*dx < ranges[n][0]:
				ranges[n] = (x - 3*dx, ranges[n][1])
			if x + 3*dx > ranges[n][1]:
				ranges[n] = (ranges[n][0], x + 3*dx)

ranges = [ranges[n] for n in elements]

# Add on the star-specific parameters
ranges = ranges + len(materials) * [(-8,0), (-3,3), (0,2000)]
ndim = len(ranges)

parameters = [[m.name + " $\log f$",m.name + " $\log \delta_X$",m.name + " $T_c (\mathrm{K})$"] for m in materials]
parameters = [p for q in parameters for p in q]
parameters = elements + parameters

# Generate masks that index into elements for each material
# so we can quickly retrieve the elements for a given material.
masks = []
for m in materials:
	mask = []
	for n in m.names:
		mask.append(elements.index(n))
	masks.append(mask)

# Priors
def Prior(cube, ranges):
	for i in range(len(cube)):
		cube[i] = cube[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0] 
	return cube

def bulkPrior(bulk):
	return np.sum(m.likelihood(bulk[mask]) for (m,mask) in zip(*(materials, masks)))

def globalPrior(params):
	# Global parameters go first, then abundances.
	bulk = params[:len(elements)]
	return bulkPrior(bulk)

# Probabilites
def model(params):
	bulk = params[:len(elements)]
	stars = [params[len(elements) + 3*i:len(elements) + 3*(i+1)] for i in range(len(materials))]	
	accrete = [bulk[mask] * m.depletedFactor(s[2], 0.01, s[1]) for (s,m,mask) in zip(*(stars, materials, masks))]
	abundances = [np.log10(10**bulk[mask] * (1-10**s[0]) + 10**accrete * 10**s[0]) for (s,accrete,mask) in zip(*(stars, accrete, masks))]
	return abundances

def globalLikelihood(params):
	abundances = model(params)
	loglike = np.sum([m.likelihood(a) for (m,a) in zip(*(materials, abundances))])
	return loglike

def probability(params):
	logp = globalLikelihood(params) + globalPrior(params)
	return logp

dirname = '../Output/Global/'

prefix = dirname + 'chain_'

def pri(x):
	return Prior(x, ranges)

result = solve(LogLikelihood=probability, Prior=pri, n_dims = ndim, importance_nested_sampling=False, resume=False,outputfiles_basename=prefix,verbose=True, n_live_points=500)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')

for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)

