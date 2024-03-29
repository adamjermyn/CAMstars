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
				fi.write(e + ',' + str(mu + 0.01*np.random.randn(1)[0]) + ',' + '0.01' + '\n')
	fi.write('Referrences:\n')
	fi.write('Test\n')
	fi.close()

### Generate depleted accreting test data
# Test should recover log10(delta) = -2, fX = 0.5 and the fAcc, which are set in increments of 0.1.

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
for j,name in enumerate(names):
	facc = 10**(-1 + 0.2 * j)
	fi = open('Data/test_accreting/' + name + '.csv','w')
	fi.write('logfAcc: ' + str(np.log10(facc)) + '\n')
	fi.write('dlogfAcc: ' + str(0.01) + '\n')
	fi.write('Abundance Normalization: H0\n')
	fi.write('Element,logX,dlogX\n')
	for e in sol.names:
		if (np.random.rand() < 0.9):
			mu,_ = sol.query(e)
			i = indicator(e)
			if i == 1:
				mu += np.log10(1 - facc + facc * 10**(-2.0))
			elif i is None:
				mu += np.log10(1 - facc + facc * (1 - 0.5 + 0.5 * 10**(-2.0)))
			if np.isfinite(mu):
				fi.write(e + ',' + str(mu + 0.01*np.random.randn(1)[0]) + ',' + '0.1' + '\n')
	fi.write('Referrences:\n')
	fi.write('Test\n')
	fi.close()