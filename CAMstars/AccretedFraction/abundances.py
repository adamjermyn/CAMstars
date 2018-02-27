import numpy as np
from CAMstars.Misc.utils import propagate_errors

def accHelper(y):
	lf = y[0]
	lx = y[1]
	lrx = y[2]

	f = 10**lf
	x = 10**lx
	r = 10**lrx

	return np.log10((1./f) * (x - r) + r)

def accretedAbundance(lf, dlf, lx, dlx, lrx, dlrx):
	'''
	Calculates the abundance of the accreting material
	given the observed abundance and a reference bulk abundance.

	Arguments:
		lf	-	log(f_accreted)
		dlf	-	Uncertainty in lf
		lx	-	Observed log(X)
		dlx	-	Uncertainty in lx
		lrx	-	Reference log(X)
		dlrx	-	Uncertainty in lrx
	'''

	lacc = accHelper((lf,lx,lrx))

	dlacc = propagate_errors(accHelper, (lf, lx, lrx), (dlf, dlx, dlrx))

	return lacc, dlacc

