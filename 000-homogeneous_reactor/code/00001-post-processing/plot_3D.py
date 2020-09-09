import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('stfs_2')

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = np.linspace(0, 2, 1000)
yline = np.linspace(0, 3, 1000)
zline = np.linspace(0, 5, 1000)

ones = np.ones(len(xline))
zeros = np.zeros(len(xline))

ax.plot3D(xline, zeros, zeros, 'gray')
ax.plot3D(ones * 2, yline, zeros, 'gray')
ax.plot3D(ones * 2, ones * 3, zline, 'gray')

ax.scatter(2, 3, 5)
ax.text(2, 3, 5, 'thermodyamic state')

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_zlim(0, 5)

ax.set_xlabel('$Y_c$')
ax.set_ylabel('$H$')
ax.set_zlabel('$Z$')

ax.set_title('Thermochemical state-space')

plt.show()