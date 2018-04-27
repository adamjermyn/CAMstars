from CAMstars.Parsers.stars import accretingPop, AJMartinPop, LFossatiPop

netPop = accretingPop + AJMartinPop + LFossatiPop
netParams = list(set(k for s in netPop.materials for k in s.params.keys() ))
netX = netPop.species

fi = open('Data/summary.csv','w+')

fi.write('Name,Source,')
for p in netParams:
	fi.write(str(p) + ',')
for x in netX:
	fi.write('log '+str(x)+',dlog '+str(x)+',')
fi.write('\n')

for s in netPop.materials:
	fi.write(s.name + ',')
	if s in accretingPop.materials:
		fi.write('Accreting Sample,')
	elif s in AJMartinPop.materials:
		fi.write('AJ Martin,')
	elif s in LFossatiPop.materials:
		fi.write('L Fossati,')
	for p in netParams:
		if p in s.params:
			fi.write(str(s.params[p]) + ',')
		else:
			fi.write(',')
	for x in netX:
		if x in s.names:
			xx, dx = s.query(x)
			fi.write(str(xx) + ',' + str(dx) + ',')
		else:
			fi.write(',,')
	fi.write('\n')

fi.close()