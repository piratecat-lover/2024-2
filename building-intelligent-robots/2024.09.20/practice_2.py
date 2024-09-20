ph = 0.0005
pfp = 0.0008
ptp = 0.75

l = ptp

pe = (l*ph) + (pfp * (1-ph))
pos = l*ph/pe
print(pos)

ph = pos
pe = (l*ph) + (pfp * (1-ph))
pos = l*ph/pe
print(pos)