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

	# Load bulk stellar properties
	params = {}
	for i,line in enumerate(fi):
		if 'Element' in line:
			break
		ind = line.index(':')
		s = (line[:ind], line[ind+1:].strip())
		try:
			params[s[0]] = float(s[1])
		except ValueError:
			params[s[0]] = s[1]

	# Load abundance data
	for i,line in enumerate(fi):
		if 'References' in line:
			break
		s = line.rstrip().split(',')
		if len(s) == 3:
			names.append(s[0])	
			logX.append(float(s[1]))
			dlogX.append(float(s[2]))

	logX = np.array(logX)
	dlogX = np.array(dlogX)

	# Calculate additional parameters
	if 'dlogmdotMinus' in params.keys() and 'dlogmdotPlus' in params.keys():
		params['dlogmdot'] = 0.5 * (params['dlogmdotMinus'] + params['dlogmdotPlus'])

	# Correct for different normalizations
	if params['Abundance Normalization'] == 'Ntot':
		nH = 1 - sum(10**w for i,w in enumerate(logX) if names[i] != 'H')
		logX = logX - np.log10(nH)
	elif params['Abundance Normalization'] == 'H12':
		logX -= 12

	# Correct for systematic uncertainties
	if 'General uncertainty' in params.keys():
		dlogX = (dlogX**2 + params['General uncertainty']**2)**0.5

	return material(name, names, logX, dlogX, params=params)

dir_path = os.path.dirname(os.path.realpath(__file__))
files = glob(dir_path + '/../../Data/Accreting Stars/*.csv')
materials = list([parse(f) for f in files])
accretingPop = population(materials)

files = glob(dir_path + '/../../Data/Field Stars/AJMartin/*.csv')
materials = list([parse(f) for f in files])
AJMartinPop = population(materials)

files = glob(dir_path + '/../../Data/Field Stars/LFossati/*.csv')
materials = list([parse(f) for f in files])
LFossatiPop = population(materials)

sol = parse(dir_path + '/../../Data/Field Stars/Sol.csv')
