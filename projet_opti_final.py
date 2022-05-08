from casadi import *
import time
import numpy as np
import matplotlib.pyplot as plt

# Question 4

## Constantes

t0 = 6 #h
tf = 19 #h
delta_t = 0.25 #h
N = int((tf-t0)/delta_t)

ALPHA = 100
k = 0.2 #h-1
T_sat = 70+273.15 #°C
T_f = 70+273.15 #°C
T_in = 50+273.15 #°C

C = 100 #C/Wh
P_M = 3*10**3 #W
t = [(t0 + i*delta_t) for i in range(N+1)] #h
Q = [0 for _ in range(N+1)] #W
Q[24] = 3
E = [2*np.exp(-(t[i]-13)**2/9) for i in range(N+1)] #kJ

## Fonctions

def h(x1, x2, alpha = ALPHA):
    return (x1*np.exp(-alpha*x1) + x2*np.exp(-alpha*x2))/(np.exp(-alpha*x1)+ np.exp(-alpha*x2))

def resol_q4(plot=False):
    opti = casadi.Opti()
    P = opti.variable(N+1)
    T = opti.variable(N+1)
    f = 0
    for i in range(1, N+1):
        f -= h(E[i], P[i]*delta_t*3.6) #tout en kJ
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
        plt.plot(t, sol.value(P)*delta_t*3.6, label = 'Energie consommée')
        plt.plot(t, E, label = 'Energie produite')
        plt.legend()
        plt.show()
        plt.plot(t, sol.value(T) - 273.15, label = 'Temperature')
        plt.show()
    return sol.value(P), sol.value(T) - 273.15

# puissance4, temperature4 = resol_q4()

# print("\nQuestion 4 - Résultats")
# print("\nPuissance consommée (en kW) :\n")
# print(' '.join(str(np.round(puissance4, 2))))
# print(f'Energie produite (en kJ):\n')
# print(' '.join(str(np.round(E, 2))))
# print("\nTempérature (en °C) :\n")
# print(' '.join(str(np.round(temperature4, 2))))



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
        opti.subject_to(s[i]-P[i]*delta_t*3.6 <= 0)
        opti.subject_to(s[i]-E[i] <= 0)
    opti.solver('ipopt');
    sol = opti.solve();
    return sol.value(P), sol.value(T)-273.15

# puissance10, temperature10 = resol_q10()

# # print("Résultats question 10\n")
# print(f" Puissance (en kW) =  {' '.join(str(np.round(puissance10, 2)))}\n")
# print(f" Température (en °C) = {' '.join(str(np.round(temperature10, 2)))}\n")
# plt.plot(t, np.round(puissance10*delta_t*3.6, 2), label = 'Energie consommée')
# plt.plot(t, E, label = 'Energie produite')
# plt.legend()
# plt.title("Graphe de l'énergie produite et de l'énergie consommée en fonction du temps (en kJ)")
# plt.show()
# plt.plot(t, temperature10, label = 'Temperature (°C)')
# plt.show()


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
        opti.subject_to(s[i]-P[i]*delta_t*3.6 <= 0)
        opti.subject_to(s[i]-E[i] <= 0)
    for i in range(n0, n0+nL):
        opti.subject_to(s[i]-P[i]*delta_t*3.6 <= 0)
        opti.subject_to(s[i]-max(0, E[i]-PL*delta_t*3.6)<=0)
    for i in range(n0+nL, N+1):
        opti.subject_to(s[i]-P[i]*delta_t*3.6 <= 0)
        opti.subject_to(s[i]-E[i] <= 0)
    opti.solver('ipopt');
    sol = opti.solve();
    return sol.value(P), sol.value(T)-273.15

# puissance13, temperature13 = resol_sous_pb(0)
# print("\nQuestion 13 - Résultats")
# print("\nPuissance (en kW) :\n")
# print(' '.join(str(np.round(puissance13, 5))))
# print("\nTempérature (en °C) :\n")
# print(' '.join(str(np.round(temperature13, 5))))

# t = np.array([k*delta_t for k in range(N+1)])
# plt.plot(t, np.round(puissance13*delta_t*3.6, 5), label = 'Puissance')
# plt.plot(t, E, label = 'Energie produite')
# plt.legend()
# plt.title("Graphe de l'énergie produite et de l'énergie consommée en fonction du temps (en kJ)")
# plt.show()
# plt.plot(t, np.round(temperature13, 5), label = 'Temperature')
# plt.title("Température du chauffe-eau en fonction du temps")
# plt.show()


## Question 14

def resol_q14():
    P_ce = []
    P_mal = []
    P =[]
    T = []
    P_cons = []
    for n in range(N-nL):
        # on calcule toutes les valeurs pour un temps de départ n
        Pi, Ti = resol_sous_pb(n)
        P_ce.append(Pi)
        pmal = [0 for _ in range(N+1)]
        for k in range(n, n+nL):
            Pi[k] += PL
            pmal[k] += PL
        P.append(Pi)
        P_mal.append(pmal)
        T.append(Ti)
        res_cons = 0
        for k in range(n):
            res_cons += min(Pi[k]*delta_t*3.6, E[k])
        for k in range(n, n+nL):
            res_cons += min((Pi[k]+PL)*delta_t*3.6, E[k])
        for k in range(nL+n, N):
            res_cons += min(Pi[k]*delta_t*3.6, E[k])
        P_cons.append(res_cons)
    i = np.argmax(P_cons)
    return P[i], P_ce[i], P_mal[i], T[i], i, P_cons

# puissance14, puissance_ce, puissance_mal, temperature14, i, P_cons = resol_q14()
# puissance_mal = np.array(puissance_mal)
# print(f' Puissance (en kW) =  {puissance14}')
# print(f' Température (en °C) = {temperature14}')
# print(f"Temps de démarrage : {i*delta_t} h")
# print(f"Puissance totale consommée à chaque instant de départ (en kW): {P_cons}")
# print(puissance_ce)
# print(puissance_mal)
# print(np.array(E)/delta_t)
# print("\nQuestion 14 - Résultats")
# print("\nTemps de départ :\n")
# print(f"{i*delta_t}h")
# print("\nPuissance totale (en kW) :\n")
# print(' '.join(str(np.round(puissance14, 2))))
# print("\nPuissance du chauffe-eau (en kW) :\n")
# print(' '.join(str(np.round(puissance_ce, 2))))
# print("\nPuissance de la machine à laver (en kW) :\n")
# print(' '.join(str(np.round(puissance_mal, 2))))
# print("\nTempérature (en °C) :\n")
# print(' '.join(str(np.round(temperature14, 2))))
# print(f'Energie totale auto-consommée : {np.round(P_cons[i], 2)} kJ\n')

# t = np.array([k*delta_t for k in range(N+1)])
# plt.plot(t, np.round(puissance14*delta_t, 2), label = 'Energie consommée totale')
# plt.plot(t, np.round(puissance_ce*delta_t*3.6, 2), label = "Energie consommée par le chauffe-eau")
# plt.plot(t, np.round(puissance_mal*delta_t*3.6, 2), label = "Energie consommée par la machine à laver")
# plt.plot(t, E, label = 'Energie produite')
# plt.legend()
# plt.title("Graphe de l'énergie produite et de l'énergie consommée\nen fonction du temps dans le cas optimal")
# plt.show()
# plt.plot(t[:N-nL], np.round(P_cons, 2))
# plt.title("Puissance auto-consommée en fonction du temps de départ")
# plt.show()
# plt.plot(t, np.round(temperature14, 2))
# plt.title("Température du chauffe-eau en fonction du temps")
# plt.show()


## Question 16

def resol_q16(P0=0, T0=T_in, s0=np.zeros((N+1),), delta0=np.zeros((N+1),), M=10**3):
    opti = casadi.Opti()
    P = opti.variable(N+1)
    T = opti.variable(N+1)
    s = opti.variable(N+1)
    delta = opti.variable(N+1)
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
    for i in range(N+1):
        opti.subject_to(s[i] - E[i] <= 0)
        opti.subject_to(s[i] - (P[i] + delta[i]*PL)*delta_t*3.6 <= 0)
    # Contraintes sur delta
    delta_sum = 0
    for i in range(N):
        delta_sum += delta[i]
        opti.subject_to(nL - delta_sum + M*(delta[i]-delta[i+1]-1) <= 0)
    for i in range(N+1):
        opti.subject_to(delta[i]**2 - delta[i] == 0)
        opti.subject_to(delta_sum - nL == 0) #pour sommer les coeffs de delta
    opti.set_initial(P, P0)
    opti.set_initial(s, s0)
    opti.set_initial(T, T0)
    opti.set_initial(delta, delta0)
    opti.solver('ipopt');
    sol = opti.solve();
    return sol.value(s), sol.value(P), sol.value(T)-273.15, np.argmax(sol.value(delta))

s, puissance16, temperature16, temps = resol_q16(10**3)
# print(s)
# print(f' Puissance (en kW) =  {puissance16}')
# print(f' Température (en °C) = {temperature16}')
# print(f"Temps de démarrage : {temps*delta_t}")

t = np.array([k*delta_t+t0 for k in range(N+1)])
plt.plot(t, np.round(puissance16*delta_t*3.6, 2), label = 'Energie consommée totale')
plt.plot(t, np.round(E, 2), label = "Energie produite")
plt.legend()
plt.title("Graphe de l'énergie produite et de l'énergie consommée\nen fonction du temps dans le cas optimal")
plt.show()
plt.plot(t, np.round(temperature16, 2))
plt.title("Température du chauffe-eau en fonction du temps")
plt.show()

# question 17 

def resol_q16(P0=0, T0=T_in, s0=np.zeros((N+1),), delta0=np.zeros((N+1),), M=10**3):
    opti = casadi.Opti()
    P = opti.variable(N+1)
    T = opti.variable(N+1)
    s = opti.variable(N+1)
    delta = opti.variable(N+1)
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
    for i in range(N+1):
        opti.subject_to(s[i] - E[i] <= 0)
        opti.subject_to(s[i] - (P[i] + delta[i]*PL)*delta_t*3.6 <= 0)
    # Contraintes sur delta
    delta_sum = 0
    for i in range(N):
        delta_sum += delta[i]
        opti.subject_to(nL - delta_sum + M*(delta[i]-delta[i+1]-1) <= 0)
    for i in range(N+1):
        opti.subject_to(delta[i]**2 - delta[i] <= 0)
        - delta[i] <= 0
        opti.subject_to(delta_sum - nL == 0) #pour sommer les coeffs de delta
    opti.set_initial(P, P0)
    opti.set_initial(s, s0)
    opti.set_initial(T, T0)
    opti.set_initial(delta, delta0)
    opti.solver('ipopt');
    sol = opti.solve();
    return sol.value(s), sol.value(P), sol.value(T)-273.15, np.argmax(sol.value(delta))

s, puissance17, temperature17, temps = resol_q16(10**3)
# print(s)
# print(f' Puissance (en kW) =  {puissance17}')
# print(f' Température (en °C) = {temperature17}')
# print(f"Temps de démarrage : {temps*delta_t}")

t = np.array([k*delta_t+t0 for k in range(N+1)])
plt.plot(t, np.round(puissance16*delta_t*3.6, 2), label = 'Energie consommée totale')
plt.plot(t, np.round(E, 2), label = "Energie produite")
plt.legend()
plt.title("Graphe de l'énergie produite et de l'énergie consommée\nen fonction du temps dans le cas optimal")
plt.show()
plt.plot(t, np.round(temperature17, 2))
plt.title("Température du chauffe-eau en fonction du temps")
plt.show()