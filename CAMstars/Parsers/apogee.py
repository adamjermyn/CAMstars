import numpy as np
#import pyfits
from astropy.io import fits
from CAMstars.Material.material import material

def checkStar(bitmask):
	'''
	Checks if there are any warnings at BAD-level on the star-level bitmask
	and returns True if so (else False). For documentation see
	http://www.sdss.org/dr14/algorithms/bitmasks/#APOGEE_STARFLAG
	'''
	if bitmask < 0:
		bitmask += 2**32
	bitstr = '{0:032b}'.format(bitmask)
	bits = list(bool(int(b)) for b in bitstr)
	checkbits = [0,3,4] # For bad-level
#	checkbits = [0,1,2,3,4,9,10,11,12,13,16,17] # For warning-level
	for i,b in enumerate(bits):
		if i in checkbits and b:
			return True
	return False

def checkASPCAP(bitmask):
	'''
	Checks if there are any warnings at BAD-level on the ASPCAP bitmask
	and returns True if so (else False). For documentation see
	http://www.sdss.org/dr14/algorithms/bitmasks/#APOGEE_STARFLAG
	'''
	if bitmask < 0:
		bitmask += 2**32
	bitstr = '{0:032b}'.format(bitmask)
	bits = list(bool(int(b)) for b in bitstr)
	ignorebits = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,28,29] # For bad-level
#	ignorebits = [12,13,15,28,29] # For warning-level
	for i,b in enumerate(bits):
		if i not in ignorebits and b:
			return True
	return False

def checkParams(bitmask):
	'''
	Checks if there are any warnings at BAD-level on the ASPCAP parameter bitmasks
	and returns True if so (else False). For documentation see
	http://www.sdss.org/dr14/algorithms/bitmasks/#APOGEE_STARFLAG
	'''
	if bitmask < 0:
		bitmask += 2**32
	bitstr = '{0:032b}'.format(bitmask)
	bits = list(bool(int(b)) for b in bitstr)
	ignorebits = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] # For bad-level
#	ignorebits = [3,4,5,6,7,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] # For warning-level
	for i,b in enumerate(bits):
		if i not in ignorebits and b:
			return True
	return False


# Open the file
fname = "/data/vault/asj42/apogee/allStar-l31c.2.fits"
fi = fits.open(fname, memmap=True)

# Read element symbols and references
dataH, headerH = fi[3].data, fi[3].header

elems = dataH.field(1)[0]
elems = list(e for e in elems if len(e) > 0)

refs = dataH.field(3)[0]

# Read data table
dataF, header = fi[1].data, fi[1].header

# Build dictionary indexing columns by keys
data = {}
for h in header.keys():
	if 'TTYPE' in h:
		col = int(h[5:]) - 1
		data[header[h]] = dataF.field(col)

# Convenience definitions
starid = data['APOGEE_ID']
teff = data['TEFF']
dteff = data['TEFF_ERR']
starflag = data['STARFLAG']
aspcapflag = data['ASPCAPFLAG']
paramflags = data['PARAMFLAG']
elemflags = data['ELEMFLAG']
numStars = len(teff)

# Build star objects

stars = []
for i in range(numStars):


	# Filter out stars with bad abundances/parameters
	# Note we use uncalibrated abundances because we need
	# to have main-sequence stars.
	bad = False
	if checkStar(starflag[i]):
		bad = True
	if checkASPCAP(aspcapflag[i]):
		bad = True
	for j in [0,3]: # We only care about Teff, M/H
		if checkParams(paramflags[i][j]):
			bad = True
	if teff[i] < -10:
		bad = True
	if bad:
		continue

	# Convert abundances into solar X/H
	felem = data['FELEM']
	df = data['FELEM_ERR']
	
	logX = []
	dlogX = []
	names = []
	for j in range(len(elems)):
		if 'I' not in elems[j] and felem[i][j] > -1000 and not checkParams(elemflags[i][j]):
			names.append(elems[j])
			if refs[j] == 0:
				# Relative to M
				logX.append(felem[j] + data['M_H'][i])
				dlogX.append((df[j]**2 + data['M_H_ERR'][i]**2)**0.5)
			else:
				# Relative to H
				logX.append(felem[j])
				dlogX.append(df[j])

	# Package
	name = starid[i]
	params = {}
	params['T'] = teff[i]
	params['dT'] = dteff[i]

	if teff[i] > 6000:
		print(name, names, params)

	stars.append(material(name, names, logX, dlogX))


