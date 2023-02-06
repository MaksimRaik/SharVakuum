import numpy as np
import matplotlib.pyplot as plt
from numba import jit

N = 1000
N_mass = np.array([4 * 10 ** (i) for i in np.arange(0, 3, 1)])
eps = 6.0 * 1.0e-3
delta_t = 1.0e-4
delta_t_mass = np.array([1.0e-8 * 10 ** (i) for i in np.arange(0, 5, 1)])
delta_m = 1.0 / N
gamma = 5.0 / 3.0
R_limit = 120.0

# характерные параметры

T0 = 3500.0  # K, Начальная температура
E0 = 3.15 * 10e9  # удельная энергия эрг/г
ro0 = 5.45 * 10e-4  # начальная плотность, г/см3
P0 = 1.15 * 10e6  # Начальное давление эрг/см3
V = (3 * ro0)

# Характерные параметры

Rx = 100.0  # Rx=R0
Ux = 5.6 * 10e4  # np.sqrt(E0/M), среднемассовая скорость
tx = 0.847 * 10e-3  # Rx/Ux, sec
rox = (1 / 3.0 * ro0)
mx = (rox * (Rx ** 3) / 3.0)
Px = ((1.0 / 3) * ro0 * E0)
Ex = 3.15 * 10e9

u = np.array([])
r = np.array([])
ro = np.array([])
V = np.array([])
P = np.array([])
E = np.array([])

uy = np.zeros( ( 3, N - 1 ) )
rx = np.zeros( ( 3, N - 1 ) )
Rhoy = np.zeros( ( 3, N - 1 ) )
Ey = np.zeros( ( 3, N - 1 ) )

#@jit
def start_values():
    global u, r, ro, V, P, E
    global uy, rx, Rhoy, Ey, k, N
    # массивы для хранения массивов для графиков
    uy = np.zeros((3, N))
    rx = np.zeros((3, N))
    Rhoy = np.zeros((3, N))
    Ey = np.zeros((3, N))

    # начальнеые условия

    u = np.zeros(N + 1)
    ro = 3.0 * np.ones(N + 1)
    V = 1 / ro
    P = ro * (gamma - 1.0)
    E = np.ones(N + 1)
    r = np.zeros(N + 1)

    k = 0

    @jit
    def r_zeros(r0):
        global delta_m, N

        for i in np.arange(1, N + 1, 1):
            r0[i] = (r0[i - 1] ** 3 + delta_m) ** (1 / 3.0)

        return r0

    r = r_zeros(r)

    # print( len( list( r ) ) )


@jit
def u_function(P1, P2, r, u):
    global delta_t, delta_m

    return u - delta_t / delta_m * r ** 2 * (P1 - P2)


@jit
def r_function(r, u):
    global delta_t

    return r + u * delta_t


@jit
def P_function(E, V):
    global gamma

    return (gamma - 1.0) * E / V


@jit
def E_function(E, V1, V2):
    global gamma

    return E * V2 / V1 * (V1 + (gamma - 1.0) / 2.0 * (V1 - V2)) / (V2 - (gamma - 1.0) / 2.0 * (V1 - V2))


@jit
def V_function(r1, r2):
    global delta_m

    return (r1 ** 3 - r2 ** 3) / 3.0 / delta_m


@jit
def cycle_for(u, r, V, E, P):

    global u_function, r_function, V_function, P_function, E_function

    global N

    for j in np.arange(0, N - 1, 1):
        u[j + 1] = u_function(P[j + 1], P[j], r[j + 1], u[j + 1])

        r[j + 1] = r_function(r[j + 1], u[j + 1])

        V_old = V[j]

        V[j] = V_function(r[j + 1], r[j])

        E[j] = E_function(E[j], V_old, V[j])

        P[j] = P_function(E[j], V[j])

    u[N] = u_function(0, P[N - 1], r[N], u[N])

    r[N] = r_function(r[N], u[N])

    V_old = V[N - 1]

    V[N - 1] = V_function(r[N], r[N - 1])

    E[N - 1] = E_function(E[N - 1], V_old, V[N - 1])

    P[N - 1] = P_function(E[N - 1], V[N - 1])

    # print( " YES from cycle_for " )

    return u, r, V, E, P


@jit
def cycle_while(u, r, V, E, P, delta_t):

    global cycle_for, massive_for_print
    global R_limit, eps, tx
    global uy, Rhoy, Ey, rx

    time = np.zeros(3)

    rm = 0

    for ii in np.arange( 0, np.int( 10.0 / delta_t ), 1 ):#while r[-1] <= R_limit:

        if r[-1] > R_limit:

            break

        u, r, V, E, P = cycle_for(u, r, V, E, P)

        if np.fabs( R_limit / 3.0 * ( rm + 1 ) - r[-1] ) <= eps:  # отбор значений для построение графиков N / step значений по времени

            uy[rm] = u[:-1]
            Rhoy[rm] = 1.0 / V[:-1]
            Ey[rm] = E[:-1]
            rx[rm] = r[:-1]

            #massive_for_print(u[:-1], r[:-1], 1.0 / V[:-1], E[:-1])

            time[ rm ] = round(ii * delta_t * tx, 4)

            rm = rm + 1

    # print( " YSE from cycle_while " )

    return time

# @jit
def plot_graph(x, y, x_label, y_label, save, Legend, log):
    global delta_t, step, MM

    plt.figure(figsize=(12, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if log == True:
        plt.yscale('log')
        # plt.xscale( 'log' )

    for i in np.arange(0, len(x), 1):
        plt.plot(x[i], y[i])

    plt.legend(Legend)
    plt.grid()
    plt.savefig(save + '.png', dpi=300)






