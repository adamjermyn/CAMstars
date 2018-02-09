import numpy as np
from utils import gaussianLogLike

# Default ranges if no prior specified.
defaultRanges = {'T':(3000,15000),'vsini':(0,500),'sini':(0,1),'M':(0.5,10),'R':(0.5,10),'log mdot':(-15,0)}
# Default errors if not specified. rel means relative, abs means absolute.
defaultSigs = {'T':(0.1,'rel'),'vsini':(0.1,'rel'),'sini':(0.1,'rel'),'M':(0.1,'rel'),'R':(0.1,'rel'),'log mdot':(0.5,'abs')}

def setupPrior(name, params):
	mu = params[0]
	sig_left = params[1]
	sig_right = params[2]

	if np.isfinite(mu):
		# Means a gaussian likelihood

		if sig_left == 0:
			ds = defaultSigs[name]
			if ds[1] == 'rel':
				sig_left = ds[0] * mu
			else:
				sig_left = ds[0]

		if sig_right == 0:
			ds = defaultSigs[name]
			if ds[1] == 'rel':
				sig_right = ds[0] * mu
			else:
				sig_right = ds[0]

		def prior(x):
			if x > mu:
				return gaussianLogLike((x - mu) / sig_right)
			else:
				return gaussianLogLike((x - mu) / sig_left)

		return prior, (mu - 3*sig_left, mu + 3*sig_right)
	else:
		# Use a uniform prior
		def prior(x):
			return 0
		return prior, defaultRanges[name]

class starPrior:
	def __init__(self, fname):
		# Load base stellar properties
		fi = open('../Data/' + fname + '_star.csv','r')
		self.properties = {}
		for line in fi:
			if '#' not in line and 'Property' not in line:
				line = line.strip().split(',')
				line = [line[0], np.array(list(map(float, line[1:-1]))), line[-1]]
				self.properties[line[0]] = line[1:]

		self.mass, mRange = setupPrior('M', self.properties['M'][0])
		self.radius, rRange = setupPrior('R', self.properties['R'][0])
		self.temperature, tRange = setupPrior('Teff', self.properties['Teff'][0])
		self.vsini, vsRange = setupPrior('vsini', self.properties['vsini'][0])
		self.sini, iRange = setupPrior('sini', self.properties['sini'][0])
		self.logmdot, mdRange = setupPrior('log mdot', self.properties['log mdot'][0])

		self.ranges = [tRange, vsRange, iRange, mRange, rRange, mdRange]

	def prior(self, params):
		m, r, t, vs, i, md = params
		return self.mass(m) + self.radius(r) + self.temperature(t) + self.vsini(vs) + self.sini(i) + self.logmdot(md)
