from casadi import *
import time
import numpy as np

# Question 4

## Constantes

t0 = 6
tf = 19
delta_t = 0.25
N = int((tf-t0)/delta_t)

ALPHA = 100
k = 0.2
T_sat = 70+273.15
T_f = 70+273.15
T_in = 50+273.15

C = 100
P_M = 3*10**3
t = [(t0 + i*delta_t) for i in range(N+1)]
Q = [0 for _ in range(N+1)]
Q[24] = 3
E = [2*np.exp(-(t[i]-13)**2/9) for i in range(N+1)]


## Fonctions

def h(x1, x2, alpha = ALPHA):
    return (x1*np.exp(-alpha*x1) + x2*np.exp(-alpha*x2))/(np.exp(-alpha*x1)+ np.exp(-alpha*x2))

opti = casadi.Opti()
n = 2*(N+1)
x = opti.variable(n)
f = 0
for i in range(1, N+1):
    f -= h(E[i], x[i]*delta_t)
opti.minimize(f)
opti.subject_to(x[N] == 0)
opti.subject_to(x[N+1]-T_in == 0)
for i in range(1, N+1):
    opti.subject_to(x[N+1+i]-np.exp(-k*delta_t)*x[N+1+i-1]-(1-np.exp(-k*delta_t))/k*C*(x[i-1]-Q[i-1])==0)
for i in range(N+1):
    opti.subject_to(-x[N+1+i]+273.15 <= 0)
    opti.subject_to(x[N+1+i]-T_sat <= 0)
    opti.subject_to(-x[i] <= 0)
    opti.subject_to(x[i] - P_M <= 0)
opti.solver('ipopt');
sol = opti.solve();
print(f' Pression =  {sol.value(x[N+1:])}')
print( f' Température = {sol.value(x[:N+1])}')

## Question 10


opti = casadi.Opti()
n = 2*(N+1)
ns = N+1
x = opti.variable(n)
s = opti.variable(ns)
f = 0
for i in range(1, N+1):
    f -= h(E[i], x[i]*delta_t)
opti.minimize(f)
opti.subject_to(x[N] == 0)
opti.subject_to(x[N+1]-T_in == 0)
for i in range(1, N+1):
    opti.subject_to(x[N+1+i]-np.exp(-k*delta_t)*x[N+1+i-1]-(1-np.exp(-k*delta_t))/k*C*(x[i-1]-Q[i-1])==0)
for i in range(N+1):
    opti.subject_to(-x[N+1+i]+273.15 <= 0)
    opti.subject_to(x[N+1+i]-T_sat <= 0)
    opti.subject_to(-x[i] <= 0)
    opti.subject_to(x[i] - P_M <= 0)
for i in range(ns):
    opti.subject_to(s[i]-x[i] <= 0)
    opti.subject_to(s[i]-E[i] <= 0)
opti.solver('ipopt');
sol = opti.solve();
print(f' Pression =  {sol.value(x[N+1:])}')
print( f' Température = {sol.value(x[:N+1])}')

