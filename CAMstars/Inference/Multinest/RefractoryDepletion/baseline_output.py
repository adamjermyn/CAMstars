
import numpy as np
from CAMstars.Parsers.stars import accretingPop, AJMartinPop, LFossatiPop, sol
from CAMstars.Material.population import population
from CAMstars.Misc.constants import mSun, yr

# Combine the field populations
field = AJMartinPop + LFossatiPop

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
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

for m in accretingPop.materials:
	if m.name in exclude_S:
		ind = m.queryIndex('S')
		if ind is not None:
			m.names.pop(ind)
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

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
				np.delete(m.logX, ind)
				np.delete(m.dlogX, ind)

for m in field.materials:
	if m.name in exclude_Zn:
		ind = m.queryIndex('Zn')
		if ind is not None:		
			m.names.pop(ind)
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

include_Na = [
'HD139614',
'HD144432'
]

for m in accretingPop.materials:
	if m.name not in include_Na:
		ind = m.queryIndex('Na')
		if ind is not None:
			m.names.pop(ind)
			np.delete(m.logX, ind)
			np.delete(m.dlogX, ind)

field = population(field.materials)

for i,s in enumerate(field.species):
	print(s, field.logX[i], field.dlogX[i])