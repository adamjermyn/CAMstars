import os
import numpy as np
from glob import glob
from CAMstars.Material.material import material
from CAMstars.Material.population import population

def parse(fname):
	fi = open(fname,'r')
	name = fname[fname.rfind('/')+1:fname.find('csv')-1]
	names = []
	logX = []
	dlogX = []

	# These objects come with a reported systematic uncertainty to apply to all elements.
	generalSigma = 0

	for i,line in enumerate(fi):
		if i == 0:
			s = line.rstrip().split(' ')
			generalSigma = float(s[-1])
		elif i > 1:
			s = line.rstrip().split(',')
			if len(s) == 3:
				names.append(s[0])
				logX.append(float(s[1]))
				dlogX.append((float(s[2])**2 + generalSigma**2)**0.5)

	# L Fossati et. al. report abundances in absolute terms rather than
	# relative to hydrogen, so we need to correct for this.
	nH = 1 - sum(10**w for w in logX)
	logX = list(w - np.log10(nH) for w in logX)


	return material(name, names, logX, dlogX)

dir_path = os.path.dirname(os.path.realpath(__file__))
files = glob(dir_path + '/../../Data/Field Stars/AJMartin/*.csv')
materials = list([parse(f) for f in files])
LFossatiPop = population(materials)