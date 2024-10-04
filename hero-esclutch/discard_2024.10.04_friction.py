g = 9.81
m = 0.1
vals = [0.604,0.252,0.3722,0.196,0.424,0.184,0.432,0.208]
real_vals = [val/m/g for val in vals]

print(real_vals)