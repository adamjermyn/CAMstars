import numpy as np
from material import material

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
