import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#Input data
x = np.array((1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5))
y = np.array((1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6))
z = np.array((42,32,20,10,0,52,40,30,20,10,62,55,50,37,20,72,69,80,52,30,82,70,60,50,40,91,82,70,60,50))

#Membuat matriks pangkat untuk orde polinom 
x_2 = np.ones(len(x))       #membuat matriks yang menampung nilai x dan ya kuadrat
y_2 = np.ones(len(x))

for i in range (0, len(x)):     #looping untuk memasukkan nilai x dan y kuadrat
    x_2[i] = x[i]**2
    y_2[i] = y[i]**2

x_2 = x_2.T         #Mentranspose matriks
y_2 = y_2.T

#mendefinisikan matriks kernel
G = np.ones((len(x),6))
x=x.T
y=y.T

for i in range (0,len(x)):      #input nilai x, x^2, y, y^2 dan xy
    G[(i,1)] = x[i]
    G[(i,2)] = x_2[i]
    G[(i,3)] = y[i]
    G[(i,4)] = y_2[i]
    G[(i,5)] = x[i]*y[i]

#menghitung Parameter model solusi    
m = (inv((G.T).dot(G))).dot(G.T).dot(z)

z_2 = G.dot(m)

#Melakukan plot grafik
a = np.arange(-10,10)
b = np.arange(-5,10)
X, Y = np.meshgrid(a,b)
Z=m[0]+m[1]*(X)+m[2]*(X**2)+m[3]*(Y)+m[4]*(Y**2)+m[5]*(X*Y)

fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
plt.suptitle('Permukaan Anomali Regional', fontsize='18')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (mGal)')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
