import numpy as np
from copy import deepcopy
from CAMstars.Parsers.stars import accretingPop, AJMartinPop, LFossatiPop, sol
from CAMstars.Material.population import population

# Combine the field populations
field = AJMartinPop + LFossatiPop

field = deepcopy(field)
accretingPop = deepcopy(accretingPop)

# Exclude elements that have unreliable error estimates

exclude_Temp = [
'UCAC12284506',
'UCAC12284746',
'UCAC12065075',
'UCAC12284653',
'UCAC12284594',
'UCAC12065058',
'V380 Ori B'
]

for m in field.materials:
	if m.name in exclude_Temp:
		field.materials.remove(m)
for m in accretingPop.materials:
	if m.name in exclude_Temp:
		accretingPop.materials.remove(m)

exclude_S = [
'HD100546', 
'HD31648',
'HD36112',
'HD68695',
'HD179218',
'HD244604',
'HD123269',
'UCAC11105213',
'UCAC11105379',
'T Ori' 
]

for m in field.materials:
	if m.name in exclude_S:
		ind = m.queryIndex('S')
		if ind is not None:			
			m.names.pop(ind)
			m.logX = np.delete(m.logX, ind)
			m.dlogX = np.delete(m.dlogX, ind)

for m in accretingPop.materials:
	if m.name in exclude_S:
		ind = m.queryIndex('S')
		if ind is not None:			
			m.names.pop(ind)
			m.logX = np.delete(m.logX, ind)
			m.dlogX = np.delete(m.dlogX, ind)
		print(m.name,m.names)

 
exclude_Zn = [
'UCAC11105106',
'UCAC11105213',
'UCAC11105379'
]

for m in accretingPop.materials:
	if m.name != 'HD144432':
		if 'Zn' in m.names:
			ind = m.queryIndex('Zn')
			if ind is not None:
				m.names.pop(ind)
				m.logX = np.delete(m.logX, ind)
				m.dlogX = np.delete(m.dlogX, ind)

for m in field.materials:
	if m.name in exclude_Zn:
		ind = m.queryIndex('Zn')
		if ind is not None:		
			m.names.pop(ind)
			m.logX = np.delete(m.logX, ind)
			m.dlogX = np.delete(m.dlogX, ind)

include_Na = [
'HD139614',
'HD144432'
]

for m in accretingPop.materials:
	if m.name not in include_Na:
		ind = m.queryIndex('Na')
		if ind is not None:
			m.names.pop(ind)
			m.logX = np.delete(m.logX, ind)
			m.dlogX = np.delete(m.dlogX, ind)

for m in field.materials:
	if m.name not in include_Na:
		ind = m.queryIndex('Na')
		if ind is not None:
			m.names.pop(ind)
			m.logX = np.delete(m.logX, ind)
			m.dlogX = np.delete(m.dlogX, ind)

# Filter out stars with no known Mdot
accretingPop = population([m for m in accretingPop.materials if 'logfAcc' in m.params.keys() and 'dlogfAcc' in m.params.keys()])

field = population(field.materials)
accretingPop = population(accretingPop.materials)
stars = accretingPop.materials
