import scipy.interpolate as sin
import numpy as np
import os

def interp(data, x0, y0):
	# Now we're going to assume that rRange and tRange are the same across all tables.
	# First, interpolate the 2D R vs. T grid across the X,Y values of interest.
	x = [i[0] for i in data]
	y = [i[1] for i in data]
	z = [i[4] for i in data]
	table = sin.griddata((x, y), z, (x0, y0))
	return table

def bilinear_interpolator(data,xPts,yPts):
	return sin.RegularGridInterpolator((xPts,yPts),data,bounds_error=False,fill_value=np.nan)

class opac:

	def __init__(self, opalName, fergName, x, y):
		self.a = opacInt(opalName, x, y, "opal")
		self.b = opacInt(fergName, x, y, "ferg")

	def opacity(self, t, rho):
		if not isinstance(t, np.ndarray):
			op = self.b.opacity(t,rho)[0,0]
			if np.isnan(op):
				op = self.a.opacity(t,rho)[0,0]
		else:
			op = self.b.opacity(t,rho)
			whereNan = np.where(np.isnan(op))
			op[whereNan] = self.a.opacity(t[whereNan[0]],rho[whereNan[1]])
		return op

	def dkdT(self,t,rho,eps=1e-3):
		k0 = 10**self.opacity(t*(1-eps),rho)
		k1 = 10**self.opacity(t*(1+eps),rho)
		return (k1-k0)/(2*t*eps)

	def dkdRho(self,t,rho,eps=1e-3):
		k0 = 10**self.opacity(t,rho*(1-eps))
		k1 = 10**self.opacity(t,rho*(1+eps))
		return (k1-k0)/(2*rho*eps)


class opacInt:

	def __init__(self, fname, x, y,opalFerg):
		self.cutoff = 0
		self.data = None
		if opalFerg=="opal":
			self.cutoff = 10
			self.data = readOpalTables(fname)
		elif opalFerg=="ferg":
			self.cutoff = 12
			self.data = readFergTables(fname)
		else:
			raise Exception("No table type specified.")
		self.interpData = interp(self.data, x, y)
		self.interpolator = bilinear_interpolator(self.interpData,self.data[0][3],self.data[0][2])

	def opacityTR(self,t,r):
		tt = np.copy(t)
		tt[tt<600] = 600.
		tt = np.log10(tt)
		r = np.log10(r)
		r = np.array(r)
		r[r>1] = 0.99
		k = self.interpolator(np.dstack((tt,r)))
		k[np.isnan(k)] = 2*self.cutoff
		k[k>self.cutoff] = np.nan # Each table has a maximum value, so this cuts off interpolation there
		return k

	def opacity(self, t, rho):
		r = rho / (t * 1e-6) ** 3
		return self.opacityTR(t, r)

# Method for reading in non-enriched OPAL tables (i.e. just X,Y,Z are
# nonzero, no dXc or dXo). Tested on latest (as of August 2014)
# GS98 composition tables.


def readOpalTables(fname):
	f = open(fname)
	# checks if we're in the zone where the tables are (as opposed to the
	# header)
	tables = False
	x = 0
	y = 0  # note that z = 1-x-y by definition
	data = []
	for line in f:
		line = line.rstrip('\n')  # remove newlines
		if tables and len(line) > 2:  # eliminates empty lines
			if 'TABLE' in line:  # reads in x,y,z for the table
				s = line.replace('=', ' ').split(' ')
				for i, a in enumerate(s):
					if a == 'X':
						x = float(s[i + 1])
					elif a == 'Y':
						y = float(s[i + 1])
				data.append([x, y, [], [], []])
			elif 'logT' in line:  # reads in the logR values for the table
				s = line.split(' ')
				rRange = [float(a) for a in s[1:] if len(a) > 0]
				data[-1][2] = rRange
			elif 'R' not in line:  # reads in the table
				s = line.split(' ')
				s = [i for i in s if len(i) > 0]
				t = float(s[0])
				s = s[1:]
				data[-1][3].append(t)
				data[-1][4].append([])
				for i, a in enumerate(s):
					data[-1][4][-1].append(float(a))
				if len(s) < len(data[-1][2]):
					for i in range(len(data[-1][2]) - len(s)):
						data[-1][4][-1].append(1e10)  # absurd value
		if '**************************' in line:
			tables = True
	for i in range(len(data)):
		data[i][4] = np.array(data[i][4])
	return data
	# Note that there is some redundancy, as rRange and tRange are expected to
	# be the same for each table. We leave the parser more general, however, as the
	# wasted space is minimal.

# Method for reading in Ferguson tables. Tested on latest (as of August 2014)
# GS98 composition tables.


def readFergTable(fname, x, y):
	# These tables are pre-split by (X,Z) value, so we can just focus on the
	# reading part.
	f = open(fname)
	# We're intentionally keeping the format the same as the opalParser format.
	data = [x, y, [], [], []]
	for line in f:
		line = line.rstrip('\n')  # remove newlines
		if 'log T' in line:  # reads in the logR values for the table
			s = line.split(' ')
			rRange = [float(a) for a in s[2:] if len(a) > 0]
			data[2] = rRange
		# reads in the table
		elif 'R' not in line and len(line) > 1 and 'Grev' not in line:
			s = line
			t = float(s[:5])
			ss = []
			counter = 6
			while counter < len(s):
				ss.append(s[counter:counter + 7])
				counter += 7
			data[3].append(t)
			data[4].append([])
			# Now we need to filter for columns which merged due to Fortran
			# formatting
			s = ss
			for i, a in enumerate(s):
				data[4][-1].append(float(a))
			if len(s) < len(data[2]):
				for i in range(len(data[2]) - len(s)):
					data[4][-1].append(1e10)  # absurd value
	data[4] = np.array(data[4])
	data[4] = data[4][::-1]
	data[3] = np.array(data[3])
	data[3] = data[3][::-1]
	return data


def readFergTables(dirName):
	data = []
	for filename in os.listdir(dirName):
		if filename[0] != '.':
			s = filename[4:]  # remove the 'g' from the beginning
			s = s.split('.')
			x = float(s[0]) / 10 ** len(s[0])
			z = float(s[1]) / 10 ** len(s[1])
			y = 1 - x - z
			data.append(readFergTable((dirName + filename), x, y))
	return data
