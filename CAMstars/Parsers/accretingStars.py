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
	paramNames = ['T','dT','M','dM','R','dR','vrot','dvrot','logmdot','dlogmdotMinus','dlogMdotPlus']
	params = {}
	for i,line in enumerate(fi):
		s = line.rstrip().split(' ')
		s[0] = s[0][:-1]
		params[s[0]] = float(s[1])
		if 'Element' in line:
			break

	# Load abundance data
	for i,line in enumerate(fi):
		s = line.rstrip().split(',')
		if len(s) == 3:
			names.append(s[0])
			logX.append(float(s[1]))
			dlogX.append(float(s[2]))

	return material(name, names, logX, dlogX, params=params)

files = glob('../../Data/Accreting Stars/*.csv')
materials = list([parse(f) for f in files])
accretingPop = population(materials)