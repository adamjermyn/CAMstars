import os
import numpy as np
from CAMstars.Misc.caseInsensitiveDict import CaseInsensitiveDict

# Load condensation temperatures
condenseTemps = CaseInsensitiveDict()
dir_path = os.path.dirname(os.path.realpath(__file__))
fi = open(dir_path + '/../../Data/Chemistry/condense.csv','r')
for i,line in enumerate(fi):
	if i > 0:
		s = line.rstrip().split(',')
		s[1] = float(s[1])
		condenseTemps[s[0]] = s[1]