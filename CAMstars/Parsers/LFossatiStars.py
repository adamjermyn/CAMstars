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
	for i,line in enumerate(fi):
		if i > 0:
			s = line.rstrip().split(',')
			if len(s) == 3:
				names.append(s[0])
				logX.append(float(s[1]))
				dlogX.append(float(s[2]))

	# L Fossati et. al. report abundances in absolute terms rather than
	# relative to hydrogen, so we need to correct for this.
	nH = 1 - sum(10**w for w in logX)
	logX = list(w - np.log10(nH) for w in logX)


	return material(name, names, logX, dlogX)

files = glob('../../Data/Field Stars/AJMartin/*.csv')
materials = list([parse(f) for f in files])
LFossatiPop = population(materials)