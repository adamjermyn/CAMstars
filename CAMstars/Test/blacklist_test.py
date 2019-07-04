from CAMstars.Parsers.pop_filter import field, accretingPop
from CAMstars.Parsers.stars import AJMartinPop, LFossatiPop
from CAMstars.Parsers.stars import accretingPop as unfiltered_accreting_pop
from CAMstars.Parsers.pop_filter import exclude_Temp, exclude_S, exclude_Zn, include_Na

unfiltered_field = AJMartinPop + LFossatiPop

# Verify that all species remaining in all materials that remain have the
# same abundances as they had in the unfiltered data.

for m in field.materials:
	for n in unfiltered_field.materials:
		if n.name == m.name:
			for s in m.names:
				assert m.query(s) == n.query(s)

for m in accretingPop.materials:
	for n in unfiltered_accreting_pop.materials:
		if n.name == m.name:
			for s in m.names:
				assert m.query(s) == n.query(s)

# Verify that the correct exclusions were made

for n in field.materials:
	assert n.names not in exclude_Temp
for n in accretingPop.materials:
	assert n.names not in exclude_Temp

for n in field.materials:
	if n.name in exclude_S:
		assert 'S' not in n.names
for n in accretingPop.materials:
	if n.name in exclude_S:
		assert 'S' not in n.names

for n in field.materials:
	if n.name in exclude_Zn:
		assert 'Zn' not in n.names
for n in accretingPop.materials:
	if n.name in exclude_Zn:
		assert 'Zn' not in n.names

for n in field.materials:
	if n.name not in include_Na:
		assert 'Na' not in n.names
for n in accretingPop.materials:
	if n.name not in include_Na:
		assert 'Na' not in n.names