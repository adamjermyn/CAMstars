from glob import glob
from CAMstars.Material.material import material

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

	return material(name, names, logX, dlogX)

sol = parse('../../Data/Field Stars/Sol.csv')