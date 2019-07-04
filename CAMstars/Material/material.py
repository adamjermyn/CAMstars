import numpy as np
from CAMstars.Misc.utils import gaussianLogLike, propagate_errors
from CAMstars.Parsers.condensation import condenseTemps
from CAMstars.AccretedFraction.star import star
from CAMstars.Misc.constants import mSun, yr, gSun, tSun

# For this we assume that the accreted material is lighter than the stellar material
def fraction(mass, radius, temperature, u_rot, logmdot, gradient=False):
	'''
	Calculates the fraction of the material in a star's photosphere
	which is due to accretion.

	mass is in solar units
	radius is in solar units
	temperature is in K
	u_rot is vsini in km/s
	logmdot is non-dimensionalised by Msun/yr
	'''

	# Convert mdot and u_rot to C.G.S.
	mdot = 10**(logmdot) * (mSun / yr)
	u_rot *= 1e5

	s = star(mass, radius, temperature)

	f = s.findF(u_rot, mdot, gradient=gradient) 

	return f

def logfrac(x):
	return np.log10(fraction(*x))

# Store material properties
class material:
	def __init__(self, name, names, logX, dlogX, params=None):
		'''
		A material is a collection of mass fraction abundances of different
		species along with uncertainties. In addition, helper methods are provided
		that enable querying for abundances by name, or for computing the likelihood
		of a given set of measured abundances.

		The arguments are:
			name 	-	The name of this material.
			names	-	The names of the species.
			logX 	-	The log10 of the mass fraction abundances relative to hydrogen.
			dlogX 	-	The uncertainty in logX, taken to be symmetric and gaussian.
			params 	-	Auxiliary parameters.

		Each argument is a list or numpy array, and they must be in corresponding order.

		In addition, this class stores the condensation temperature associated with each
		species. This is obtained from the condensation parser.

		'''

		self.name = name
		self.names = names
		self.logX = logX
		self.dlogX = dlogX

		if params is None:
			params = {}
		self.params = params

		if 'logL' in params:
			self.params['L'] = 10**self.params['logL']

		# If two of T, R, L are present but not the third fill in the blanks
		if 'T' in params and 'R' in params and 'L' not in params:
			calc = lambda x: (x[0]/tSun)**4 * x[1]**2
			w = (params['T'],params['R'])
			self.params['L'] = calc(w)

			if 'dT' in params and 'dR' in params:
				dw = (params['dT'],params['dR'])
				self.params['dL'] = propagate_errors(calc, w, dw)
		elif 'T' in params and 'L' in params and 'R' not in params:
			calc = lambda x: (x[0]/tSun)**(-2)*x[1]**(0.5)
			w = (params['T'],params['L'])
			self.params['R'] = calc(w)

			if 'dT' in params and 'dL' in params:
				dw = (params['dT'],params['dL'])
				self.params['dR'] = propagate_errors(calc, w, dw)
		elif 'R' in params and 'L' in params and 'T' not in params:
			calc = lambda x: tSun * (x[1] / x[0]**2)**0.25
			w = (params['R'],params['L'])
			self.params['T'] = calc(w)

			if 'dR' in params and 'dL' in params:
				dw = (params['dR'],params['dL'])
				self.params['dT'] = propagate_errors(calc, w, dw)

		# If possible, calculate log10(g)
		if 'M' in params and 'R' in params:
			gcalc = lambda x: np.log10(gSun * x[0] / x[1]**2)
			w = (params['M'], params['R'])
			self.params['logg'] = gcalc(w)

			if 'dM' in params and 'dR' in params:
				dw = (params['dM'], params['dR'])
				self.params['dlogg'] = propagate_errors(gcalc, w, dw)

		# If possible, calculate fraction of material arising from accretion.
		if 'M' in params and 'R' in params and 'T' in params and 'vrot' in params and 'logmdot' in params:
			w = (params['M'], params['R'], params['T'], params['vrot'], params['logmdot'])
			self.params['logfAcc'] = logfrac(w)

			# Calculate uncertainty in f
			if 'dM' in params and 'dR' in params and 'T' in params and 'dvrot' in params and 'dlogmdot' in params:
				dw = (params['dM'], params['dR'], params['dT'], params['dvrot'], params['dlogmdot'])
				self.params['dlogfAcc'] = propagate_errors(logfrac, w, dw)

		self.temps = np.array([condenseTemps[name] for name in self.names])


	def query(self, name):
		'''
		Returns the abundance and uncertainty associated with the specified name.
		'''

		try:
			i = next(i for i,v in enumerate(self.names) if v.lower() == name.lower()) # Case-insensitive search
			return self.logX[i], self.dlogX[i]
		except StopIteration:
			return None

	def queryIndex(self, name):
		'''
		Returns the index of the specified name.
		'''
		
		try:
			i = next(i for i,v in enumerate(self.names) if v.lower() == name.lower()) # Case-insensitive search
			return i
		except StopIteration:
			return None

	def likelihood(self, abundances, errs=None):
		'''
		Returns the gaussian log likelihood associated with the specified abundances
		relative to those in this material. If the abundances have uncertainties
		these must be provided in the optional errs argument, otherwise they are taken
		to be zero.
		'''
		
		if errs is None:
			errs = np.zeros(len(abundances))

		differences = self.logX - abundances
		combinedVariance = errs**2 + self.dlogX**2
		return np.sum(gaussianLogLike(differences / combinedVariance**0.5), axis=0)

	def __eq__(self, other):
		'''
		Overloads equality operator to be based on name.
		'''

		return self.name == other.name

	def __hash__(self):
		'''
		Overloads hashing operator to be based on name.
		'''

		return hash(self.name)

