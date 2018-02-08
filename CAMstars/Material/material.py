import numpy as np
from CAMstars.Misc.utils import gaussianLogLike

# Store material properties
class material:
	def __init__(self, logX, dlogX, names):
		'''
		A material is a collection of mass fraction abundances of different
		species along with uncertainties. In addition, helper methods are provided
		that enable querying for abundances by name, or for computing the likelihood
		of a given set of measured abundances.

		The arguments are:
			logX 	-	The log10 of the mass fraction abundances relative to hydrogen.
			dlogX 	-	The uncertainty in logX, taken to be symmetric and gaussian.
			names	-	The names of the species.

		Each argument is a list or numpy array, and they must be in corresponding order.
		'''

		self.names = names
		self.logX = logX
		self.dlogX = dlogX

	def query(self, name):
		'''
		Returns the abundance and uncertainty associated with the specified name.
		'''

		try:
			i = self.names.index(name)
			return self.logX[i], self.dlogX[i]
		except ValueError:
			return None

	def queryIndex(self, name):
		'''
		Returns the index of the specified name.
		'''
		
		try:
			i = self.names.index(name)
			return i
		except ValueError:
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