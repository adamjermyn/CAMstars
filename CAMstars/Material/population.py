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
		# We do this in an unweighted way because the weighted calculations tend to
		# give enormous weight to individual stars with (possibly) understated uncertainties.
		self.logX = np.zeros(len(self.species))
		self.dlogX = np.zeros(len(self.species))
		for i,s in enumerate(self.species):
			mats = self.speciesDict[s]
			if len(mats) > 1:
				lx, dlx = zip(*[m.query(s) for m in mats])
				lx = np.array(lx)
				dlx = np.array(dlx)
				self.logX[i] = np.average(lx)
				self.dlogX[i] = np.average((lx - self.logX[i])**2)**0.5
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
			ind = next(i for i,v in enumerate(self.species) if v.lower() == name.lower()) # Case-insensitive search
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
