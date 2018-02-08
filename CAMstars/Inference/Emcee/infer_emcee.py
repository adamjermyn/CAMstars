import numpy as np
from scipy.special import expit
import emcee
import h5py

condenseTemps = {}
fi = open('../Data/condense.csv','r')
for i,line in enumerate(fi):
	if i > 0:
		s = line.rstrip().split(',')
		s[1] = float(s[1])
		condenseTemps[s[0]] = s[1]

pref = -0.5*np.log(2*np.pi)
def gaussianLogLike(x):
	return pref - x**2

class material:
	def __init__(self, fname):
		fi = open('../Data/' + fname + '.csv','r')
		self.names = []
		self.X = []
		self.dX = []
		for i,line in enumerate(fi):
			if i > 0:
				s = line.rstrip().split(',')
				self.names.append(s[0])
				self.X.append(float(s[1]))
				self.dX.append(float(s[2]))
		self.X = np.array(self.X)
		self.dX = np.array(self.dX)
		self.temp = np.array([condenseTemps[name] for name in self.names])

	def likelihood(self, abundances):
		return np.sum(gaussianLogLike((self.X[np.newaxis,:] - abundances) / self.dX[np.newaxis,:]), axis=1)

	def depletedFactor(self, tCut, dT, delta):
		# The amount by which accreting material is enhanced for the elements
		# present in this type of material.

		df = 1 + (10**delta - 1) * expit((self.temp[:,np.newaxis] - tCut[np.newaxis,:]) * dT)
		return df

def masker(mat1, mat2):
	# Masks the abundances on material mat1 so as to match which ones are measured for mat2.
	# Note that this modifies mat1.
	nameDict = {n:(x,dx,t) for n,x,dx,t in zip(*(mat1.names, mat1.X, mat1.dX, mat1.temp))}
	mat1Xnew = np.array([nameDict[n][0] for n in mat2.names])
	mat1dXnew = np.array([nameDict[n][1] for n in mat2.names])
	mat1TempNew = np.array([nameDict[n][2] for n in mat2.names])

	mat1.names = mat2.names
	mat1.X = mat1Xnew
	mat1.dX = mat1dXnew
	mat1.temp = mat1TempNew

# Our parameter space is composed of:
#	- Bulk composition of the star
#	- Composition of the accreting material
#	- A few global parameters, namely accreted fraction, depletion factor, and possibly a few others.

def bulkPrior(sol, bulk):
	# We take as our prior on the star's bulk composition the Sun's likelihood.
	# The log prior over abundances is then sol.likelihood(bulk).
	return sol.likelihood(bulk)

def accretePrior(sol, accrete, tCut, dT, delta):
	# For the accreted material we take the same prior, but for all non-H/non-He elements
	# we assume that they are depleted by an amount that depends on their condensation temperature.
	return sol.likelihood(accrete - np.log10(sol.depletedFactor(tCut, dT, delta)).T)

def fPrior(f):
	# f represents the log of the fraction and ranges from -8 to 0.
	ret = np.zeros(f.shape)
	ret[(f<-8) | (f > 0)] = -np.inf
	return ret

def tCutPrior(tCut):
	# Uniform from 0 to 2000K
	ret = np.zeros(tCut.shape)
	ret[(tCut<0) | (tCut > 2000)] = -np.inf
	return ret

def dTPrior(dT):
	# Uniform from +-1/5 to infinity in 1/dT
	ret = np.zeros(dT.shape)
	ret[(dT<-5) | (dT > 5)] = -np.inf
	return ret

def deltaPrior(delta):
	# Uniform in log-space. Delta is a logarithmic quantity (i.e. log of enhancement),
	# so we just make this uniform. In this case from 0 to 3 (e.g. 1 to 1000).
	ret = np.zeros(delta.shape)
	ret[(delta<-3) | (delta > 3)] = -np.inf
	return ret

def globalPrior(params, sol):
	# Global parameters go first, then abundances.
	f = params[:,0]
	delta = params[:,1]
	tCut = params[:,2]
	dT = params[:,3]
	bulk = params[:,4:len(sol.X) + 4]
	accrete = params[:,4 + len(sol.X):2*len(sol.X) + 4]
	return fPrior(f) + tCutPrior(tCut) + dTPrior(dT) + deltaPrior(delta) + bulkPrior(sol, bulk) + accretePrior(sol, accrete, tCut, dT, delta)

def model(params, star):
	f = params[:,0]
	f[f>0] = 0
	bulk = params[:,4:len(star.X) + 4]
	accrete = params[:,4 + len(star.X):2*len(star.X) + 4]
	abundance = np.log10(10**bulk * (1-10**f[:,np.newaxis]) + 10**accrete * 10**f[:,np.newaxis])
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

	ndim = len(sol.X) + len(star.X) + 4	
	nwalkers = 5*ndim+2

	defaults = [-2,0,1000,1] + list(sol.X) + list(sol.X)
	scatter = [2,1,300,1] + list(sol.dX) + list(sol.dX)
	ranges = [(-8,0),(-3,3),(0,2000),(-5,5)]
	pos = [defaults + scatter*np.random.randn(ndim) for _ in range(nwalkers)]
	pos = np.array(pos)
	for i in range(len(ranges)):
		pos[pos[:,i] < ranges[i][0],i] = ranges[i][0]
		pos[pos[:,i] > ranges[i][1],i] = ranges[i][1]

	sampler = emcee.EnsembleSampler(nwalkers, ndim, probability, args=(star,sol), vectorize=True, moves=[emcee.moves.WalkMove(), emcee.moves.StretchMove()], threads=1)
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, probability, args=(hd100546,), backend=emcee.backends.HDFBackend('sampler.hdf',name='samples',read_only=False), threads=4)

	import corner
	import matplotlib.pyplot as plt
	plt.style.use('ggplot')

	for i in range(1000):
		print(i)
		pos, prob, state = sampler.run_mcmc(pos, 1000)
		samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
		mean = np.percentile(samples, 50, axis=0)

#		fig = corner.corner(samples, labels=["$\log f$","$\log \delta_X$","$T_c (\mathrm{K})$","$(\delta T/\mathrm{K})^{-1} $"] + sol.names + sol.names,quantiles=[0.16, 0.5, 0.84],show_titles=True)
#		fig.savefig("../Output/" + starName + "_triangle_full.pdf")
#		plt.close('all')

		samples = samples[:,:4]
		fig = corner.corner(samples, labels=["$\log f$","$\log \delta_X$","$T_c (\mathrm{K})$","$(\delta T/\mathrm{K})^{-1} $"] + sol.names + sol.names,quantiles=[0.16, 0.5, 0.84],show_titles=True,range=((-8,0),(-3,3),(0,2000),(-5,5)))
		fig.savefig("../Output/" + starName + "_triangle.pdf")
		plt.close('all')

		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		m = model(mean[np.newaxis,:], star)[0]
		ax.scatter(range(len(m)), m - sol.X,c='b')
		ax.errorbar(range(len(m)),star.X - sol.X,yerr=star.dX, fmt='o',c='r')
		ax.set(xticks=range(len(m)), xticklabels=star.names)
		ax.set_ylabel('$\log [X] - \log [X]_\mathrm{solar}$')
		plt.savefig('../Output/' + starName + '_model.pdf')

		plt.close('all')

import sys
run(sys.argv[1])
