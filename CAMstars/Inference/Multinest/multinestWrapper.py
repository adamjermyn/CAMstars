import os
import json
import numpy as np
import pymultinest
from pymultinest.solve import solve
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def Prior(cube, ranges):
	'''
	Converts coordinates drawn from the unit cube, given as cube, into coordinates drawn from the
	parameter cube, with dimensions specified by the corresponding entries in ranges, each of which
	is a tuple of the form (min, max).
	'''
	for i in range(len(cube)):
		cube[i] = cube[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0] 
	return cube

def run(outputDirectory, outputPrefix, ranges, parameters, loglikelihood, n_live_points=500):
	'''
	Wrapper for PyMultiNest. The arguments are:

		outputDirectory	-	The directory in which to place output files.
		outputPrefix	-	The prefix for all output file names.
		ranges			-	The ranges of all parameters. This is a list of tuples, each of which
							is of the form (min, max).
		parameters		-	The names of all parameters. This is a list whose entries correspond to those
							in ranges.
		loglikelihood 	-	A function which returns the log likelihood value for the model given parameter
							values.
		n_live_points	-	The number of live points to use while sampling.

	There is no return value, as all output is written to files.
	'''


	prefix = outputDirectory + '/' + outputPrefix

	if not os.path.exists(outputDirectory):
		os.makedirs(outputDirectory)

	ndim = len(ranges)

	def pri(cube):
		return Prior(cube, ranges)


	print(prefix)
	result = solve(LogLikelihood=loglikelihood, Prior=pri, n_dims = ndim, importance_nested_sampling=False,resume=False,outputfiles_basename=prefix,verbose=True, n_live_points=n_live_points, evidence_tolerance=0.2)

	print()
	print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
	print()
	print('parameter values:')

	for name, col in zip(parameters, result['samples'].transpose()):
		print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

	with open('%sparams.json' % prefix, 'w') as f:
		json.dump(parameters, f, indent=2)

	parameters = json.load(open(prefix + 'params.json'))
	n_params = len(parameters)

	a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
	s = a.get_stats()

	json.dump(s, open(prefix + 'stats.json', 'w'), indent=4)

def analyze(inputDirectory, inputPrefix, outputDirectory, outputPrefix):
	'''
	Helper function for analyzing distributions from PyMultiNest. 

	The arguments are:

		inputDirectory	-	The directory containing the PyMultiNest output files to process.
							This should be the 'outputDirectory' that was used to generate those files.
		inputPrefix		-	The prefix for all input file names.
							This should be the 'outputPrefix' that was used to generate those files.
		outputDirectory	-	The directory in which to place output files.
		outputPrefix	-	The prefix for all output file names.

	The return value is a PyMultiNest Analyzer object.
	'''
	prefix = inputDirectory + '/' + inputPrefix
	outputPrefix = outputDirectory + '/' + outputPrefix
	parameters = json.load(open(prefix + 'params.json'))
	n_params = len(parameters)

	a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
	s = a.get_stats()

	json.dump(s, open(outputPrefix + 'stats.json', 'w'), indent=4)

	print('  marginal likelihood:')
	print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
	print('  parameters:')
	meds = []
	for p, m in zip(parameters, s['marginals']):
		lo, hi = m['1sigma']
		med = m['median']
		meds.append(med)
		sigma = (hi - lo) / 2
		print(sigma)
		i = max(0, int(-np.floor(np.log10(sigma))) + 1)
		fmt = '%%.%df' % i
		fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
		print(fmts % (p, med, sigma))

	return a, meds

def plot1D(a, parameters, outputDirectory, outputPrefix):
	'''
	Helper function for plotting 1D distributions from PyMultiNest. 

	The arguments are:

		a 				-	A PyMultiNest Analyzer object.
		parameters		-	A list containing the names of all parameters.
		outputDirectory	-	The directory in which to place output files.
		outputPrefix	-	The prefix for all output file names.

	There is no return value, as all output is written to files.
	'''

	prefix = outputDirectory + '/' + outputPrefix

	n_params = len(parameters)
	p = pymultinest.PlotMarginal(a)
	values = a.get_equal_weighted_posterior()
	s = a.get_stats()
	modes = s['modes']

	pp = PdfPages(prefix + 'marg1d.pdf')
	
	for i in range(n_params):
		plt.figure(figsize=(3, 3))
		plt.xlabel(parameters[i])
		plt.locator_params(nbins=5)
		
		m = s['marginals'][i]
		iqr = m['q99%'] - m['q01%']
		xlim = m['q01%'] - 0.3 * iqr, m['q99%'] + 0.3 * iqr
		#xlim = m['5sigma']
		plt.xlim(xlim)
	
		oldax = plt.gca()
		x,w,patches = oldax.hist(values[:,i], bins=np.linspace(xlim[0], xlim[1], 20), edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
		oldax.set_ylim(0, x.max())
	
		newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
		p.plot_marginal(i, ls='-', color='blue', linewidth=3)
		newax.set_ylim(0, 1)
	
		ylim = newax.get_ylim()
		y = ylim[0] + 0.05*(ylim[1] - ylim[0])
		center = m['median']
		low1, high1 = m['1sigma']
		#print center, low1, high1
		newax.errorbar(x=center, y=y,
			xerr=np.transpose([[center - low1, high1 - center]]), 
			color='blue', linewidth=2, marker='s')
		oldax.set_yticks([])
		newax.set_ylabel("Probability")
		ylim = oldax.get_ylim()
		newax.set_xlim(xlim)
		oldax.set_xlim(xlim)
		plt.savefig(pp, format='pdf', bbox_inches='tight')
		plt.close()
	pp.close()

def plot2D(a, parameters, outputDirectory, outputPrefix, nbins = 20):
	'''
	Helper function for plotting 2D distributions from PyMultiNest. 

	The arguments are:

		a 				-	A PyMultiNest Analyzer object.
		parameters		-	A list containing the names of all parameters.
		outputDirectory	-	The directory in which to place output files.
		outputPrefix	-	The prefix for all output file names.
		nbins 			-	The number of bins to use when plotting. Default is 20.

	There is no return value, as all output is written to files.
	'''

	prefix = outputDirectory + '/' + outputPrefix

	n_params = len(parameters)
	p = pymultinest.PlotMarginal(a)
	values = a.get_equal_weighted_posterior()
	s = a.get_stats()
	modes = s['modes']

	plt.figure(figsize=(5*n_params, 5*n_params))
	for i in range(n_params):
		plt.subplot(n_params, n_params, i + 1)
		plt.xlabel(parameters[i])
	
		m = s['marginals'][i]
		plt.xlim(m['5sigma'])
	
		oldax = plt.gca()
		x,w,patches = oldax.hist(values[:,i], bins=nbins, edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
		oldax.set_ylim(0, x.max())
	
		newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
		p.plot_marginal(i, ls='-', color='blue', linewidth=3)
		newax.set_ylim(0, 1)
	
		ylim = newax.get_ylim()
		y = ylim[0] + 0.05*(ylim[1] - ylim[0])
		center = m['median']
		low1, high1 = m['1sigma']
		#print(center, low1, high1)
		newax.errorbar(x=center, y=y,
			xerr=np.transpose([[center - low1, high1 - center]]), 
			color='blue', linewidth=2, marker='s')
		oldax.set_yticks([])
		#newax.set_yticks([])
		newax.set_ylabel("Probability")
		ylim = oldax.get_ylim()
		newax.set_xlim(m['5sigma'])
		oldax.set_xlim(m['5sigma'])
		#plt.close()
	
		for j in range(i):
			plt.subplot(n_params, n_params, n_params * (j + 1) + i + 1)
			p.plot_conditional(i, j, bins=20, cmap = plt.cm.gray_r)
			for m in modes:
				plt.errorbar(x=m['mean'][i], y=m['mean'][j], xerr=m['sigma'][i], yerr=m['sigma'][j])
			plt.xlabel(parameters[i])
			plt.ylabel(parameters[j])
			#plt.savefig('cond_%s_%s.pdf' % (params[i], params[j]), bbox_tight=True)
			#plt.close()

	plt.savefig(prefix + 'marg.pdf')
	plt.savefig(prefix + 'marg.png')
	plt.close()









