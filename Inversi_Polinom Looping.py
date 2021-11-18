import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#Input data
x = np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
y = np.array((2,4,4.5,6,8,9,11,13,14,14,13,10,7,4,3))

#mendefinisikan matriks kernel
G = np.ones((len(x),2))

for i in range (0,len(x)):
    G[(i,1)] = x[i]
    
#menghitung Parameter model solusi    
m = (inv((G.T).dot(G))).dot(G.T).dot(y)

y_1 = G.dot(m)

#Melakukan plot grafik
xplot=np.array(range(0,len(x)+2))
yplot=np.ones(len(xplot))

for i in range (0,len(x)+2):
    yplot[i]=m[0]+xplot[i]*m[1]
    
plt.plot(xplot, yplot, '-r', label='Regresi Linear')
plt.plot(x,y, 'og', label='Data')
plt.legend(loc='lower right')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Melakukan perhitungan misfit data
sum_e = 0

for i in range (0,len(x)):
    e = (y[i]-y_1[i])**2
    sum_e = sum_e+e

print('Error data diperoleh sebagai berikut :', sum_e)

n = 2
while sum_e > 0.1 :
    
    G_new=np.ones((len(x),n))
    for i in range (0,n):
        for j in range (0,len(x)):
            G_new[(j,i)] = x[j]**i
            
    m_new = (inv(((G_new).T).dot(G_new))).dot((G_new).T).dot(y)

    y_new = G_new.dot(m_new)
    
    plt.plot(x, y_new, '-r', label='Garis Polinomial')
    plt.plot(x,y, 'og', label='Data')
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    sum_e = 0
    for i in range (0,len(x)):
        e = (y[i]-y_new[i])**2
        sum_e = sum_e+e    

    print('\nError untuk polinom orde', n, 'adalah ', sum_e)
    
    if sum_e>0.1:
        print('Akan dilakukan perhitungan polinomial orde', n+1)
    
    n=n+1    
    
print('\n\nError untuk orde polinom ke',n-1,' telah minimum :', sum_e)