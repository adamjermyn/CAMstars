from numpy import pi

c       = 2.99792458e10  		# cm/s           Speed of light
sigma   = 5.670400e-5    		# erg/cm^2/s/K^4 Stephan-Boltzmann
a       = 4*sigma/c      		# erg/cm^3/K^4   Photon gas internal energy constant
kB      = 1.3806488e-16  		# erg/K          Boltzmann
mP      = 1.672622e-24   		# g              Proton mass
newtonG = 6.67e-8        		# erg cm/g^2     G
mSun    = 1.9988435e33   		# g              Solar Mass
rSun    = 6.96e10        		# cm             Solar Radius
lSun    = 3.846e33       		# erg/s          Solar Luminosity
fSun	= lSun / (4*pi*rSun**2)		# erg/cm^2/s	 Solar surface flux
gSun    = newtonG * mSun / rSun**2 # cm/s^2		Solar surface gravity
kappaG  = 1000.         		# g/cm^2         Gamma ray opacity
yr	= 365.254*24*3600		# s		  Seconds in a year
tSun	= 5770.		 		# K		  Stellar photosphere temperature
