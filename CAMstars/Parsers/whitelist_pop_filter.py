from copy import deepcopy
from CAMstars.Parsers.pop_filter import field, accretingPop, stars
from CAMstars.Material.population import population
from CAMstars.Parsers.stars import sol
import numpy as np

field = deepcopy(field)
accretingPop = deepcopy(accretingPop)
stars = deepcopy(stars)
sol = deepcopy(sol)

# Filter out all but whitelisted elements
whitelist = ['Fe','Mg','Si','Ti','H','He','O','Zn','Na','S']
for m in accretingPop.materials:
	toremove = set(m.names).difference(set(whitelist))
	for n in toremove:
		ind = m.queryIndex(n)
		m.names.pop(ind)
		m.logX = np.delete(m.logX, ind)
		m.dlogX = np.delete(m.dlogX, ind)
for m in field.materials:
	toremove = set(m.names).difference(set(whitelist))
	for n in toremove:
		ind = m.queryIndex(n)
		m.names.pop(ind)
		m.logX = np.delete(m.logX, ind)
		m.dlogX = np.delete(m.dlogX, ind)

field = population(field.materials)
accretingPop = population(accretingPop.materials)

filtered_sol = population([sol])

for m in filtered_sol.materials:
	toremove = set(m.names).difference(set(whitelist))
	for n in toremove:
		ind = m.queryIndex(n)
		m.names.pop(ind)
		m.logX = np.delete(m.logX, ind)
		m.dlogX = np.delete(m.dlogX, ind)

filtered_sol = population(filtered_sol.materials)