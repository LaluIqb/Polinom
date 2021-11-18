import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#Input data
x = np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
y = np.array((2,3,4.5,6,8,9,11,13,14,14,13,10,7,4,3))

#Matriks baru berisi pembulatan y
y_1=np.ones(len(x))
for i in range (0,len(x)):
    y_1[i]=round(y[i])

#Membuat matriks pangkat untuk orde polinom 
x_2 = np.ones(len(x))
x_3 = np.ones(len(x))
x_4 = np.ones(len(x))
x_5 = np.ones(len(x))

for i in range (0, len(x)):
    x_2[i] = x[i]**2
    x_3[i] = x[i]**3
    x_4[i] = x[i]**4
    x_5[i] = x[i]**5


x_2 = x_2.T
x_3 = x_3.T
x_4 = x_4.T
x_5 = x_5.T

#mendefinisikan matriks kernel
G = np.ones((len(x),6))

for i in range (0,len(x)):
    G[(i,1)] = x[i]
    G[(i,2)] = x_2[i]
    G[(i,3)] = x_3[i]
    G[(i,4)] = x_4[i]
    G[(i,5)] = x_5[i]

#menghitung Parameter model solusi    
m = (inv((G.T).dot(G))).dot(G.T).dot(y)

y_2 = G.dot(m)

#Melakukan plot grafik
xplot=np.array(range(0,len(x)+2))
yplot=np.ones(len(xplot))

for i in range (0,len(x)+2):
    yplot[i]=m[0]+xplot[i]*m[1]+(xplot[i]**2)*m[2]+(xplot[i]**3)*m[3]+(xplot[i]**4)*m[4]+(xplot[i]**5)*m[5]
    
plt.plot(xplot, yplot, '-r', label='Regresi Linear')
plt.plot(x,y_1, 'og', label='Data')
plt.legend(loc='lower right')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()

sum_e = 0
for i in range (0,len(x)):
    e = (y[i]-y_2[i])**2
    sum_e = sum_e+e  
    
print('Error untuk polinom orde 5 adalah ', str(sum_e))