import numpy as np
from material import material

class population:
	def __init__(self, materials):
		'''
		A population is a collection of materials, along with helper methods that
		enable easier population-level analysis.
		'''
		
		self.materials = materials