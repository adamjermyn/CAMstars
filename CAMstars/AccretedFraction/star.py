import numpy as np
from numpy import pi
from scipy.optimize import newton

import CAMstars.Parsers.opacity as opacity
from CAMstars.Misc.constants import kB, mP, rSun, mSun, tSun, yr, newtonG, fSun, lSun, sigma

opalName = '../../Data/Opacity/Opal/GS98.txt'
fergName = '../../Data/Opacity/Ferguson/f05.gs98/'
op = opacity.opac(opalName, fergName, 0.7, 0.28) # Solar composition

def rhoFromPT(t, p, mu):
	'''
	This method returns rho given t, p, and mu.
	'''
	return mu*mP*p/(kB*t)

def approxSAHA(t):
	'''
	This implements a very crude model of the SAHA equation
	for the mean molecular weight. The model assumes half ionisation
	at 8500K, exponential behaviour, and a width of 500K.
	'''
	mu = 1.0 - 0.5/(1 + np.exp(-(t-8500)/500))
	return mu

def pFromKappa(kappa, g, tau):
	return tau*g/kappa

def findRho(op, t, g, tau=2./3):
	'''
	This method takes as input an opacity object, a temperature,
	and a surface gravity and determines the density at which tau
	matches the specified value (default is 2./3).
	'''

	# Assuming ionised
	mu = approxSAHA(t)

	def f(x):
		kappa = 10**op.opacity(t, 10**x)
		p = pFromKappa(kappa, g, tau)
		return np.log10(rhoFromPT(t, p, mu=mu)) - x

	logRho = newton(f, -6)

	return 10**logRho

class star:
	def __init__(self, mass, radius=None, temperature=None):
		# Raw properties

		self.mass = mass * mSun
		self.radius = radius * rSun
		self.temperature = temperature

		# Calculated properties

		self.area = 4 * pi * self.radius**2
		self.gravity = newtonG * self.mass / self.radius**2
		self.flux = sigma*self.temperature**4
		self.luminosity = self.area * self.flux

		# Photosphere properties

		self.density = findRho(op, self.temperature, self.gravity)
		self.opacity = 10**op.opacity(self.temperature, self.density)
		self.pressure = pFromKappa(self.opacity, self.gravity, 2./3)
		self.photoMass = (2./3) * 4 * np.pi * self.radius**2 / self.opacity
		self.height = self.pressure / (self.density * self.gravity)
		self.soundspeed = (5*self.pressure/(3*self.density))**0.5
		self.thermalDiff = self.height * self.flux / self.pressure

		# Molecular viscosity
		self.dM = 5e-5*(self.temperature/1e4)**(5./2)*(self.density/0.1)**(-1)

		# Convective viscosity (added to molecular)
		if self.mass < 1.5*mSun:
			vc = (self.flux / self.density)**(1./3)
			self.dM += vc * self.height

	def findF(self, u_rot, mdot, gradient=False, weightRatio=55.):
		# The weight ratio is the ratio of the accreting to native molecular weight.
		# For iron (weight 55, 55/2 for singly-ionised) accreting into ionised hydrogen (1/2)
		# this is 55.
		d = self.dM + self.rotationalMixing(u_rot)
		if not gradient:
			return (self.height**2 / d) * (mdot / self.photoMass)
		else:
			a = self.gradientMixing(weightRatio)
			b = d
			c = -self.height**2 * (mdot / self.photoMass)
			return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)	

	def rotationalMixing(self, u_rot):
		eps = u_rot**2/(self.radius * self.gravity)
		v = eps * (self.flux / self.pressure) * (self.height / self.radius) # The last term is to get the radial velocity
		if isinstance(v, np.ndarray):
			v[v > self.soundspeed] = self.soundspeed
		else:
			v = min(v, self.soundspeed)

		drot = v * self.height
		return drot

	def gradientMixing(self, weightRatio):
		return 1e3*self.thermalDiff * weightRatio



