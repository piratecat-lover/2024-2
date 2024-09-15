import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

def Randomwalk1D(n):
    x, y = 0,0
    xpos = [0]
    ypos = [0]
    for i in range(n):
        step = np.random.choice(['up', 'down'])
        if step == 'up':
            y += 1
        else:
            y -= 1
        x += 1
        xpos.append(x)
        ypos.append(y)
    return xpos, ypos

randwalk1D = Randomwalk1D(1000)
plt.plot(randwalk1D[0], randwalk1D[1], 'r-', label= '1D Random Walk')
plt.title("1D Random walk")
plt.show()

n = 1000
x = np.zeros(n)
y = np.zeros(n)
direction=["NORTH"," SOUTH","EAST","WEST "]
for i in range(n):
    step = random.choice(direction)
    if step == "EAST":
        x[i] = x[i - 1] + 1
        y[i] = y[i - 1]
    elif step == "WEST":
        x [i] = x [i - 1] - 1
        y [i] = y [i - 1]
    elif step == "NORTH ":
        x [i] = x [i - 1]
        y [i] = y [i - 1] + 1
    else:
        x[i] = x[i - 1]
        y[i] = y[i - 1] - 1
    pylab.title("2D Random Walk")
    pylab.plot(x, y)
    pylab.show()