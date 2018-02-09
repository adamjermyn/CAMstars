import numpy as np
from pymultinest.solve import solve
import h5py
import json
import sys
import os
from glob import glob
from material import material
from utils import gaussianLogLike

# Collect field stars
files = glob('../Data/Field/AJMartin/*.csv')
files = [f for f in files if 'condense' not in f and 'star' not in f and 'Sol' not in f]
materials = [material(f) for f in files]

# Fix for weird data format
for m in materials:
	m.X = m.X - 12

# Collect more field stars
files = glob('../Data/Field/LFossati/*.csv')
files = [f for f in files if 'condense' not in f and 'star' not in f and 'Sol' not in f]

# Identify relevant elements
materials2 = [material(f) for f in files]

# Fix for weird data format
for m in materials2:
	nH = 1 - sum(10**m.X)
	m.X -= np.log10(nH)
	m.dX = (m.dX**2 + 0.25**2)**0.5 # Accounts for overall uncertainty.
					# To do this correctly we should account for uncertainty varying by star, but this is good enough.	

materials = materials + materials2

# Identify relevant elements
elements = list(set([n for m in materials for n in m.names ]))

# Identify which stars hold which elements
stars = {e:list(s for s in materials if e in s.names) for e in elements}

# Compute abundance distributions
X = np.array(list(np.average([s.query(e)[0] for s in stars[e]], weights=[s.query(e)[1]**2 for s in stars[e]]) for e in elements))
dX = np.array(list(np.var([s.query(e)[0] for s in stars[e]]) for e in elements)) # Not quite right because this neglects our weighting, but close enough

# Priors
def Prior(cube, ranges):
	for i in range(len(cube)):
		cube[i] = cube[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0] 
	return cube

def probability(params, diff, var, s):
	logf, logd, tc = params
	f = 10**logf

	q = list([np.log((1-f) + f * 10**(logd * (s.temp[j] > tc)))  for j,e in enumerate(s.names) if e in elements])
	
	return np.sum(gaussianLogLike((q[i] - diff[i])**2/var[i]) for i in range(len(q)))

def run(fname):
	star = material(fname)

	# Star-specific parameters
	ranges = [(-3,0), (-3,3), (0,2000)]
	ndim = len(ranges)

	parameters = [star.name + " $\log f$",star.name + " $\log \delta_X$",star.name + " $T_c (\mathrm{K})$"]


	dirname = '../Output/Global/Individual/'

	prefix = dirname + 'chain_' + star.name + '_'

	def pri(x):
		return Prior(x, ranges)

	diff = list([star.X[i] - X[elements.index(e)] for i,e in enumerate(star.names) if e in elements])
	var = list([dX[elements.index(e)]**2 + star.dX[i]**2 for i,e in enumerate(star.names) if e in elements])

	def prob(x):
		return probability(x, diff, var, star)

	result = solve(LogLikelihood=prob, Prior=pri, n_dims = ndim, importance_nested_sampling=False, resume=False,outputfiles_basename=prefix,verbose=True, n_live_points=2000)

	print()
	print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
	print()
	print('parameter values:')

	for name, col in zip(parameters, result['samples'].transpose()):
		print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

	with open('%sparams.json' % prefix, 'w') as f:
		json.dump(parameters, f, indent=2)

# Collect population data files
files = glob('../Data/*.csv')
files = [f for f in files if 'condense' not in f and 'star' not in f and 'Sol' not in f]
for f in files:
	run(f)


