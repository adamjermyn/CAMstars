from CAMstars.Parsers.stars import sol
import numpy as np

### Generate field test data

numStars = 10

names = [str(i) for i in range(numStars)]
for name in names:
	fi = open('Data/test_field/' + name + '.csv','w')
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

### Generate depleted accreting test data
# Test should find that fully-refractory elements (fixedList1) are ~2.0dex depleted, hence
# log10(delta) = -2.0. The partially-refractory ones (freeList) should be found to be ~0.25dex depleted, 
# meaning that fX = 0.9/0.99 ~ 0.91.
# The remainder should be at solar abudnance.

fixedList1 = ['Fe','Mg','Si','Ti']
fixedList0 = ['H','He']
freeList = ['C','O','Zn','Na','S']

def indicator(x):
	if x in fixedList1:
		return 1
	elif x in fixedList0:
		return 0
	else:
		return None

numStars = 5

names = [str(i) for i in range(numStars)]
for name in names:
	fi = open('Data/test_accreting/' + name + '.csv','w')
	fi.write('Abundance Normalization: H0\n')
	fi.write('Element,logX,dlogX\n')
	for e in sol.names:
		if (np.random.randint(2)):
			mu,_ = sol.query(e)
			i = indicator(e)
			if i == 1:
				mu -= 2.0
			elif i is None:
				mu -= 1.0
			if np.isfinite(mu):
				fi.write(e + ',' + str(mu + 0.1*np.random.randn(1)[0]) + ',' + '0.1' + '\n')
	fi.write('Referrences:\n')
	fi.write('Test\n')
	fi.close()