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
files = files[:15]

# Identify relevant elements
materials = [material(f) for f in files]
elements = list(set([n for m in materials for n in m.names ]))

# Identify which stars hold which elements
stars = {e:list(s for s in materials if e in s.names) for e in elements}
temps = list(stars[e][0].temp[stars[e][0].queryIndex(e)] for e in elements)

# Construct prior range
# Our parameter space is composed of:
#	- For each star: accreted fraction, depletion factor, critical temperature
#
# Parameters are given in the order listed above.

# Star-specific parameters
ranges = len(materials) * [(-3,0), (-3,3), (0,2000)]
ndim = len(ranges)

parameters = [[m.name + " $\log f$",m.name + " $\log \delta_X$",m.name + " $T_c (\mathrm{K})$"] for m in materials]
parameters = [p for q in parameters for p in q]

# Precompute parts of the likelihood function
aj = [sum([s.dX[s.queryIndex(e)]**(-2) for s in stars[e]]) for e in elements]
aj = np.array(aj)

# Priors
def Prior(cube, ranges):
	for i in range(len(cube)):
		cube[i] = cube[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0] 
	return cube

# Probabilites
def probability(params):
	logf = params[::3]
	logd = params[1::3]
	tc = params[2::3]

	f = 10**logf
	ttc = np.meshgrid()

	q = [[np.log((1-f[i]) + f[i] * np.exp(logd[i] * (m.temp[j] > tc[i])))  for j,e in enumerate(m.names)] for i,m in enumerate(materials)]

	bj = [sum([s.dX[s.queryIndex(e)]**(-2) * 2 * (q[materials.index(s)][s.queryIndex(e)] - s.X[s.queryIndex(e)]) for s in stars[e]]) for e in elements]
	cj = [sum([s.dX[s.queryIndex(e)]**(-2) * (q[materials.index(s)][s.queryIndex(e)] - s.X[s.queryIndex(e)])**2 for s in stars[e]]) for e in elements]
	bj = np.array(bj)
	cj = np.array(cj)

	return -0.5 * np.sum(cj - bj**2 / (4 * aj))

dirname = '../Output/Global/'

prefix = dirname + 'chain_'

def pri(x):
	return Prior(x, ranges)

#print(probability(np.random.randn(ndim)))

#exit()

result = solve(LogLikelihood=probability, Prior=pri, n_dims = ndim, importance_nested_sampling=False, resume=False,outputfiles_basename=prefix,verbose=True, n_live_points=500)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')

for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)

