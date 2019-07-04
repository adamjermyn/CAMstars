from CAMstars.Parsers.stars import sol, parse
import numpy as np
import random
import string
import os
from glob import glob

### Generate field test data
# Fills non-abundance fields with random data then verifies that the parser
# correctly reads this data back in.

numStars = 10

fields = [''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) for _ in range(10)]
vals = np.random.randn(len(fields))
valdict = {f:v for f,v in zip(*(fields, vals))}

names = [str(i) for i in range(numStars)]
for name in names:
	fi = open('Data/test_field/' + name + '.csv','w')

	for f,v in zip(*(fields,vals)):
		fi.write(f + ': ' + str(v) + '\n')

	fi.write('Abundance Normalization: H0\n')
	fi.write('Element,logX,dlogX\n')
	for e in sol.names:
		if (np.random.randint(2)):
			mu,_ = sol.query(e)
			if np.isfinite(mu):
				fi.write(e + ',' + str(mu + 0.1*np.random.randn(1)[0]) + ',' + '0.1' + '\n')
	fi.write('Referrences:\n')
	fi.write('Test\n')
	fi.close()

dir_path = os.path.dirname(os.path.realpath(__file__))
files = glob(dir_path + '/../../Data/test_field/*.csv')
materials = list([parse(f) for f in files])

for m in materials:
	for f in fields:
		assert m.params[f] == valdict[f]

print('Success!')