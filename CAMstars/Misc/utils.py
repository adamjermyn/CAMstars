import numpy as np

# For computing gaussian priors
pref = -0.5*np.log(2*np.pi)
def gaussianLogLike(x):
	# To handle distributions with different left vs. right errors
	# we just compute x as |y - y0|/(dy_right) for y > y0  and
	# |y - y0|/(dy_left) for y < y0.
	return pref - x**2

def propagate_errors(func, x, dx, eps=1e-3):
	'''
	The arguments are:
		func 	-	The function with which to propagate errors. May take a multidimensional
					input.
		x 		-	The point at which to propagate errors. May be multidimensional.
		dx 		-	The standard deviation of x. May be multidimensional.
		eps 	-	A small value to use in computing derivatives. Default is 1e-3.

	The return value is the variance of the function value at the specified point.
	It is assumed that there are no correlations amongst the entries in x.
	'''

	dim = len(x)

	var = 0
	for i in range(dim):
		xNew = np.copy(x)

		xNew[i] *= 1 + eps
		f1 = func(xNew)
		xNew[i] /= 1 + eps
		xNew[i] *= 1 - eps
		f2 = func(xNew)

		df = (f1 - f2) / 2

		var += (dx[i] * df)**2

	return var