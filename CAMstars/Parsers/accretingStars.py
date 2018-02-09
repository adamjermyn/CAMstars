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

	# Load bulk stellar properties
	params = {}
	for i,line in enumerate(fi):
		s = line.rstrip().split(' ')
		if 'Element' in line:
			break
		s[0] = s[0][:-1]
		params[s[0]] = float(s[1])

	# Calculate additional parameters
	print(params)
	params['dlogmdot'] = 0.5 * (params['dlogmdotMinus'] + params['dlogmdotPlus'])

	# Load abundance data
	for i,line in enumerate(fi):
		s = line.rstrip().split(',')
		if len(s) == 3:
			names.append(s[0])
			logX.append(float(s[1]))
			dlogX.append(float(s[2]))

	return material(name, names, logX, dlogX, params=params)

dir_path = os.path.dirname(os.path.realpath(__file__))
files = glob(dir_path + '/../../Data/Accreting Stars/*.csv')
materials = list([parse(f) for f in files])
accretingPop = population(materials)