import numpy as np

# Load condensation temperatures
condenseTemps = {}
fi = open('../../Data/Chemistry/condense.csv','r')
for i,line in enumerate(fi):
	if i > 0:
		s = line.rstrip().split(',')
		s[1] = float(s[1])
		condenseTemps[s[0]] = s[1]