from casadi import *
import time
import numpy as np
import matplotlib.pyplot as plt

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

def resol_q4(plot=False):
    opti = casadi.Opti()
    P = opti.variable(N+1)
    T = opti.variable(N+1)
    f = 0
    for i in range(1, N+1):
        f -= h(E[i], P[i]*delta_t)
    opti.minimize(f)
    opti.subject_to(P[N] == 0)
    opti.subject_to(T[0]-T_in == 0)
    for i in range(1, N+1):
        opti.subject_to(T[i]-np.exp(-k*delta_t)*T[i-1]-(1-np.exp(-k*delta_t))/k*C*(P[i-1]-Q[i-1])==0)
    for i in range(N+1):
        opti.subject_to(-T[i]+273.15 <= 0)
        opti.subject_to(T[i]-T_sat <= 0)
        opti.subject_to(-P[i] <= 0)
        opti.subject_to(P[i] - P_M <= 0)
    opti.solver('ipopt');
    sol = opti.solve();
    if plot:
        t = np.array([k*delta_t for k in range(N+1)])
        plt.plot(t, sol.value(P), label = 'Puissance')
        plt.plot(t, E, label = 'Energie produite')
        plt.legend()
        plt.show()
        plt.plot(t, sol.value(T) - 273.15, label = 'Temperature')
        plt.show()
    return sol.value(P), sol.value(T) - 273.15

# puissance4, temperature4 = resol_q4()
# print(f' Puissance (en W) =  {puissance4}')
# print( f' Température (en °C) = {temperature4}')


## Question 10

def resol_q10():
    opti = casadi.Opti()
    ns = N+1
    P = opti.variable(N+1)
    T = opti.variable(N+1)
    s = opti.variable(N+1)
    f = 0
    for i in range(1, N+1):
        f -= s[i]
    opti.minimize(f)
    opti.subject_to(P[N] == 0)
    opti.subject_to(T[0]-T_in == 0)
    for i in range(1, N+1):
        opti.subject_to(T[i]-np.exp(-k*delta_t)*T[i-1]-(1-np.exp(-k*delta_t))/k*C*(P[i-1]-Q[i-1])==0)
    for i in range(N+1):
        opti.subject_to(-T[i]+273.15 <= 0)
        opti.subject_to(T[i]-T_sat <= 0)
        opti.subject_to(-P[i] <= 0)
        opti.subject_to(P[i] - P_M <= 0)
    for i in range(ns):
        opti.subject_to(s[i]-P[i] <= 0)
        opti.subject_to(s[i]-E[i] <= 0)
    opti.solver('ipopt');
    sol = opti.solve();
    return sol.value(P), sol.value(T)-273.15

# puissance10, temperature100 = resol_q10()
# print(f' Puissance (en W) =  {puissance10}')
# print(f' Température (en °C) = {temperature10}')


## Question 13

# Constantes
t_i0 = 0
n0 = int(t_i0/delta_t)
nL = 6
PL = 0.25


# Résolution sous-problème
def resol_sous_pb(n0):
    opti = casadi.Opti()
    P = opti.variable(N+1)
    T = opti.variable(N+1)
    s = opti.variable(N+1)
    f = 0
    for i in range(1, N+1):
        f -= s[i]
    opti.minimize(f)
    opti.subject_to(P[N] == 0)
    opti.subject_to(T[0]-T_in == 0)
    for i in range(1, N+1):
        opti.subject_to(T[i]-np.exp(-k*delta_t)*T[i-1]-(1-np.exp(-k*delta_t))/k*C*(P[i-1]-Q[i-1])==0)
    # Contraintes sur T
    for i in range(N+1):
        opti.subject_to(-T[i]+273.15 <= 0)
        opti.subject_to(T[i]-T_sat <= 0)
    # Contraintes sur P
    for i in range(N+1):
        opti.subject_to(-P[i] <= 0)
        opti.subject_to(P[i] - P_M <= 0)
    # Contraintes sur s
    for i in range(n0):
        opti.subject_to(s[i]-P[i] <= 0)
        opti.subject_to(s[i]-E[i] <= 0)
    for i in range(n0, n0+nL):
        opti.subject_to(s[i]-P[i] <= 0)
        opti.subject_to(s[i]-max(0, E[i]-PL*delta_t)<=0)
    for i in range(n0+nL, N+1):
        opti.subject_to(s[i]-P[i] <= 0)
        opti.subject_to(s[i]-E[i] <= 0)
    opti.solver('ipopt');
    sol = opti.solve();
    return sol.value(P), sol.value(T)-273.15

# puissance13, temperature13 = resol_sous_pb(0)
# print(f' Puissance (en W) =  {puissance13}')
# print(f' Température (en °C) = {temperature13}')


## Question 14

def resol_q14():
    P = []
    T = []
    P_cons = []
    for n in range(N-nL):
        Pi, Ti = resol_sous_pb(n)
        P.append(Pi)
        T.append(Ti)
        res_cons = 0
        for k in range(n0):
            res_cons += min(Pi[k], E[k])
        for k in range(n0, n0+n):
            res_cons += min(Pi[k]+PL, E[k])
        for k in range(n0+n, N):
            res_cons += min(Pi[k]*delta_t, E[k])
        P_cons.append(res_cons)
    i = np.argmax(P_cons)
    return P[i], T[i], i

puissance, temperature, i = resol_q14()
print(f' Puissance (en W) =  {puissance}')
print(f' Température (en °C) = {temperature}')
print(f"Temps de démarrage : {i*delta_t}")
    
