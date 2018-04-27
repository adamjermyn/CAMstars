import numpy as np
from CAMstars.Material.material import material

class population:
	def __init__(self, materials):
		'''
		A population is a collection of materials, along with helper methods that
		enable easier population-level analysis.
		'''

		self.materials = materials

		# Ordered list of all species in this population. Ordering is arbitrary.
		self.species = list(set(n for m in self.materials for n in m.names))

		# Dictionary with species as keys and all associated materials as values.
		self.speciesDict = {n:list(m for m in materials if n in m.names) for n in self.species}

		# We want to calculate the mean and standard deviation for the population.
		# These are stored in numpy arrays indexed in the same order as species.
		# For the mean we weight measurements by the reciprocal of their variance.
		# For the variance we use the relaibility-weighted unbiased estimator, given by
		# sum_i (x_i - mu)^2 / dx_i^2 / (V_1 - V_2 / V_1),
		# where mu is the estimated mean, V_1 = sum_i 1/dx_i^2, and V_2 = sum_i 1/dx_i^4.
		# This is as per the GSL Manual: https://www.gnu.org/software/gsl/manual/html_node/Weighted-Samples.html
		# Note that these aggregates are only meaningful if the materials of interest were
		# truly drawn from the same population.
		self.logX = np.zeros(len(self.species))
		self.dlogX = np.zeros(len(self.species))
		for i,s in enumerate(self.species):
			mats = self.speciesDict[s]
			if len(mats) > 1:
				lx, dlx = zip(*[m.query(s) for m in mats])
				lx = np.array(lx)
				dlx = np.array(dlx)
#				self.logX[i] = np.average(lx, weights=1/dlx**2)
				self.logX[i] = np.average(lx)
				self.dlogX[i] = np.average((lx - self.logX[i])**2)
#				self.dlogX[i] = np.average((lx - self.logX[i])**2, weights=1/dlx**2)
#				v1 = np.sum(1/dlx**2)
#				v2 = np.sum(1/dlx**4)
#				self.dlogX[i] /= (v1 - v2 / v1)
			else:
				# Otherwise there's no population to aggregate.
				self.logX[i], self.dlogX[i] = mats[0].query(s)


	def __add__(self, other):
		'''
		Combines two populations.
		'''

		# Filter out overlap between populations.
		materials = list(set(self.materials + other.materials))

		# Notify user if there is overlap.
		if len(materials) != len(self.materials) + len(other.materials):
			print('NOTE: Overlapping materials detected.')

		return population(materials)

	def __len__(self):
		'''
		Returns the number of materials in the population.
		'''
		return len(self.materials)

	def queryStats(self, name):
		'''
		Returns the population mean and standard deviation for the specified species.
		'''
		if name in self.species:
			ind = self.species.index(name)
			return self.logX[ind], self.dlogX[ind]
		else:
			return None

	def query(self, name):
		'''
		Returns a list of tuples, each of which contains the name of a
		material containing the specified species along with the abundance and
		its associated uncertainty.
		'''

		ret = []
		for m in self.materials:
			q = m.query(name)
			if q is not None:
				ret.append((m.name, q[0], q[1]))

		return ret
