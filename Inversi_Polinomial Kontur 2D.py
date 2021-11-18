import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#Input data
x = np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
y = np.array((3.5,5,6.5,8,9.5,11,12.5,14,15.5,17,18.5,20,21.5,23,24.5))
z = np.arange(10,85,5)

#Membuat matriks pangkat untuk orde polinom 
x_2 = np.ones(len(x))
y_2 = np.ones(len(x))

for i in range (0, len(x)):
    x_2[i] = x[i]**2
    y_2[i] = y[i]**2

x_2 = x_2.T
y_2 = y_2.T

#mendefinisikan matriks kernel
G = np.ones((len(x),5))
x=x.T
y=y.T

for i in range (0,len(x)):
    G[(i,1)] = x[i]
    G[(i,2)] = x_2[i]
    G[(i,3)] = y[i]
    G[(i,4)] = y_2[i]

#menghitung Parameter model solusi    
m = (inv((G.T).dot(G))).dot(G.T).dot(z)

z_2 = G.dot(m)

#Melakukan plot grafik
a = b = np.arange(1,20)
X, Y = np.meshgrid(a,b)
Z=m[0]+m[1]*(X)+m[2]*(X**2)+m[3]*(Y)+m[4]*(Y**2)
plt.contourf(X, Y, Z)

plt.show()
