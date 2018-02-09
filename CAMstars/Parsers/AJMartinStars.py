import os
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

	# AJ Martin et. al. report abundances offset by +12, so we need
	# to correct for this.
	logX = list([w - 12 for w in logX])

	return material(name, names, logX, dlogX)

dir_path = os.path.dirname(os.path.realpath(__file__))
files = glob(dir_path + '/../../Data/Field Stars/AJMartin/*.csv')
materials = list([parse(f) for f in files])
AJMartinPop = population(materials)