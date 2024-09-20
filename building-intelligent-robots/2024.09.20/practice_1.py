ph = 0.00545
pfp = 0.000922456
ptp = 0.033898305

l = ptp
pe = (l*ph) + (pfp * (1-ph))

pos = l*ph/pe

print(pos)

ph = pos
pe = (l*ph) + (pfp * (1-ph))

pos = l*ph/pe

print(pos)