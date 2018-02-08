import numpy as np

# For computing gaussian priors
pref = -0.5*np.log(2*np.pi)
def gaussianLogLike(x):
	# To handle distributions with different left vs. right errors
	# we just compute x as |y - y0|/(dy_right) for y > y0  and
	# |y - y0|/(dy_left) for y < y0.
	return pref - x**2

