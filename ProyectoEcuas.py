"""
Autores:
Bryan Martínez 23542
Javier Cifuentes 23079
Adriana Palacios 23044
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#---------------------------------------------------------------- Algoritmos -------------------------------------------------------------------

def adams_moulton(funcion, x0, y0, h, pasos):
    """
    Método de Adams-Moulton para EDOs de primer orden y sistemas.

    Parámetros:
    - funcion: Función que describe la EDO o sistema de EDOs.
    - x0: Valor inicial de x.
    - y0: Valor inicial de y (puede ser escalar o vector).
    - h: Tamaño de paso.
    - pasos: Número de pasos a realizar.
    """
    if np.isscalar(y0):
        y0 = np.array([y0])
    
    num_ecuaciones = len(y0)
    x_vals = np.zeros(pasos + 1)
    y_vals = np.zeros((pasos + 1, num_ecuaciones))
    x_vals[0] = x0
    y_vals[0] = y0

    for i in range(pasos):
        x_i = x_vals[i]
        y_i = y_vals[i]
        
        # Predicción inicial usando Euler
        y_pred = y_i + h * funcion(x_i, y_i)
        
        # Actualizamos el siguiente valor de x
        x_vals[i+1] = x_i + h
        
        # Corrección usando Adams-Moulton
        tolerancia = 1e-6
        max_iteraciones = 10
        iteracion = 0
        y_nuevo = y_i + (h/2) * (funcion(x_i, y_i) + funcion(x_vals[i+1], y_pred))
        
        while np.linalg.norm(y_nuevo - y_pred) > tolerancia and iteracion < max_iteraciones:
            y_pred = y_nuevo
            y_nuevo = y_i + (h/2) * (funcion(x_i, y_i) + funcion(x_vals[i+1], y_pred))
            iteracion += 1
        
        y_vals[i+1] = y_nuevo
    
    return x_vals, y_vals

def runge_kutta_4(funcion, x0, y0, h, pasos):
    """
    Método de Runge-Kutta de 4to Orden (RK4) para EDOs de primer orden y sistemas.

    Parámetros:
    - funcion: Función que describe la EDO o sistema de EDOs.
    - x0: Valor inicial de x.
    - y0: Valor inicial de y (puede ser escalar o vector).
    - h: Tamaño de paso.
    - pasos: Número de pasos a realizar.
    """
    if np.isscalar(y0):
        y0 = np.array([y0])
    
    num_ecuaciones = len(y0)
    x_vals = np.zeros(pasos + 1)
    y_vals = np.zeros((pasos + 1, num_ecuaciones))
    x_vals[0] = x0
    y_vals[0] = y0

    for i in range(pasos):
        x_i = x_vals[i]
        y_i = y_vals[i]
        
        # Cálculo de las pendientes k1, k2, k3, k4
        k1 = funcion(x_i, y_i)
        k2 = funcion(x_i + h/2, y_i + h/2 * k1)
        k3 = funcion(x_i + h/2, y_i + h/2 * k2)
        k4 = funcion(x_i + h, y_i + h * k3)
        
        # Actualización del valor de y
        y_nuevo = y_i + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Actualizamos el siguiente valor de x y y
        x_vals[i+1] = x_i + h
        y_vals[i+1] = y_nuevo
    
    return x_vals, y_vals

#---------------------------------------------------------------- ------------ -------------------------------------------------------------------

#----------------------------------- EDO de Primer Orden -----------------------------------
def edo_primer_orden():
    # Ecuación de primer orden: y' = (3x^3 - y) / (x * ln(x))
    # Condición inicial: y(e) = 1
    # Solución analítica: y = x^3 / ln(x)
    
    def funcion(x, y):
        return (3 * x**3 - y[0]) / (x * np.log(x))
    
    def solucion_analitica(x):
        return (x**3 + (1 - np.exp(3)))/ np.log(x)
    
    x0 = math.e
    y0 = 1
    h = 0.01
    pasos = 10000

    x_vals_am, y_vals_am = adams_moulton(funcion, x0, y0, h, pasos)
    x_vals_rk4, y_vals_rk4 = runge_kutta_4(funcion, x0, y0, h, pasos)
    y_exacta = solucion_analitica(x_vals_am)

    # Calcular el error relativo
    error_relativo_am = np.abs((y_vals_am[:, 0] - y_exacta) / y_exacta)
    error_relativo_rk4 = np.abs((y_vals_rk4[:, 0] - y_exacta) / y_exacta)

    # Calcular el error medio relativo
    error_medio_relativo_am = np.mean(error_relativo_am)
    error_medio_relativo_rk4 = np.mean(error_relativo_rk4)
    print("EDO de Primer Orden:")
    print("Error medio relativo Adams-Moulton:", error_medio_relativo_am)
    print("Error medio relativo RK4:", error_medio_relativo_rk4)

    # Graficar las soluciones
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals_am, y_vals_am[:, 0], label="Adams-Moulton", linestyle='-', color='blue')
    plt.plot(x_vals_rk4, y_vals_rk4[:, 0], label="RK4", linestyle='--', color='green')
    plt.plot(x_vals_am, y_exacta, label="Solución Analítica", linestyle=':', color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("EDO de Primer Orden: Comparación de Soluciones")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_am, error_medio_relativo_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo')
    plt.title('EDO de Primer Orden: Comparación de Errores Medios Relativos')
    plt.show()

#----------------------------------- EDO de Segundo Orden -----------------------------------
def edo_segundo_orden():
    # Ecuación de segundo orden: y'' - 2y' + y = x e^x + 4
    # Condiciones iniciales: y(0) = 1, y'(0) = 1
    # Solución analítica: y = (1/6)x^3 e^x + x e^x + e^x + 4

    def funcion(x, y):
        y1 = y[0]
        y2 = y[1]
        dy1_dx = y2
        dy2_dx = 2 * y2 - y1 + x * np.exp(x) + 4
        return np.array([dy1_dx, dy2_dx])
    
    def solucion_analitica(x):
        return (1/6)*x**3 * np.exp(x) + x * np.exp(x) + np.exp(x) + 4
    
    x0 = 0
    y0 = np.array([1, 1])  # y(0) = 1, y'(0) = 1
    h = 0.01
    pasos = 10000

    x_vals_am, y_vals_am = adams_moulton(funcion, x0, y0, h, pasos)
    x_vals_rk4, y_vals_rk4 = runge_kutta_4(funcion, x0, y0, h, pasos)
    y_exacta = solucion_analitica(x_vals_am)

    # Calcular el error relativo
    error_relativo_am = np.abs((y_vals_am[:, 0] - y_exacta) / y_exacta)
    error_relativo_rk4 = np.abs((y_vals_rk4[:, 0] - y_exacta) / y_exacta)

    # Calcular el error medio relativo
    error_medio_relativo_am = np.mean(error_relativo_am)
    error_medio_relativo_rk4 = np.mean(error_relativo_rk4)
    print("Error medio relativo Adams-Moulton:", error_medio_relativo_am)
    print("Error medio relativo RK4:", error_medio_relativo_rk4)

    # Graficar las soluciones
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals_am, y_vals_am[:, 0], label="Adams-Moulton", linestyle='-', color='blue')
    plt.plot(x_vals_rk4, y_vals_rk4[:, 0], label="RK4", linestyle='--', color='green')
    plt.plot(x_vals_am, y_exacta, label="Solución Analítica", linestyle=':', color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("EDO de Segundo Orden: Comparación de Soluciones")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_am, error_medio_relativo_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo')
    plt.title('EDO de Segundo Orden: Comparación de Errores Medios Relativos')
    plt.show()

#----------------------------------- Sistema de EDOs 2x2 -----------------------------------
def sistema_edo_2x2():
    # Sistema de ecuaciones diferenciales:
    # x' = 4x + 5y
    # y' = -2x + 6y
    # Condiciones iniciales: x(0) = 1, y(0) = 1

    def funcion(t, y):
        x = y[0]
        y_var = y[1]
        dx_dt = 4 * x + 5 * y_var
        dy_dt = -2 * x + 6 * y_var
        return np.array([dx_dt, dy_dt])
    
    def solucion_analitica(t):
        x_exacta = (1/5)*np.exp(5*t)*(5*np.cos(3*t) + 4*np.sin(3*t))
        y_exacta = (1/5)*np.exp(5*t)*(np.cos(3*t) - 3*np.sin(3*t)) + (4/15)*np.exp(5*t)*(3*np.cos(3*t) + np.sin(3*t))
        return x_exacta, y_exacta
    
    t0 = 0
    y0 = np.array([1, 1])
    h = 0.01
    pasos = 10000

    t_vals_am, y_vals_am = adams_moulton(funcion, t0, y0, h, pasos)
    t_vals_rk4, y_vals_rk4 = runge_kutta_4(funcion, t0, y0, h, pasos)
    x_exacta_am, y_exacta_am = solucion_analitica(t_vals_am)

    # Calcular el error relativo para x(t)
    error_relativo_x_am = np.abs((y_vals_am[:, 0] - x_exacta_am) / x_exacta_am)
    error_relativo_x_rk4 = np.abs((y_vals_rk4[:, 0] - x_exacta_am) / x_exacta_am)

    # Calcular el error relativo para y(t)
    error_relativo_y_am = np.abs((y_vals_am[:, 1] - y_exacta_am) / y_exacta_am)
    error_relativo_y_rk4 = np.abs((y_vals_rk4[:, 1] - y_exacta_am) / y_exacta_am)

    # Calcular el error medio relativo para x(t) y y(t)
    error_medio_relativo_x_am = np.mean(error_relativo_x_am)
    error_medio_relativo_x_rk4 = np.mean(error_relativo_x_rk4)
    error_medio_relativo_y_am = np.mean(error_relativo_y_am)
    error_medio_relativo_y_rk4 = np.mean(error_relativo_y_rk4)
    print("Error medio relativo en x(t) Adams-Moulton:", error_medio_relativo_x_am)
    print("Error medio relativo en x(t) RK4:", error_medio_relativo_x_rk4)
    print("Error medio relativo en y(t) Adams-Moulton:", error_medio_relativo_y_am)
    print("Error medio relativo en y(t) RK4:", error_medio_relativo_y_rk4)

    # Graficar las soluciones para x(t)
    plt.figure(figsize=(12, 6))
    plt.plot(t_vals_am, y_vals_am[:, 0], label="Adams-Moulton x(t)", linestyle='-', color='blue')
    plt.plot(t_vals_rk4, y_vals_rk4[:, 0], label="RK4 x(t)", linestyle='--', color='green')
    plt.plot(t_vals_am, x_exacta_am, label="Solución Analítica x(t)", linestyle=':', color='red')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Sistema de EDOs 2x2: Solución para x(t)")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras para x(t)
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_x_am, error_medio_relativo_x_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo en x(t)')
    plt.title('Sistema de EDOs 2x2: Comparación de Errores Medios Relativos en x(t)')
    plt.show()
    
    # Graficar las soluciones para y(t)
    plt.figure(figsize=(12, 6))
    plt.plot(t_vals_am, y_vals_am[:, 1], label="Adams-Moulton y(t)", linestyle='-', color='blue')
    plt.plot(t_vals_rk4, y_vals_rk4[:, 1], label="RK4 y(t)", linestyle='--', color='green')
    plt.plot(t_vals_am, y_exacta_am, label="Solución Analítica y(t)", linestyle=':', color='red')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.title("Sistema de EDOs 2x2: Solución para y(t)")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras para y(t)
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_y_am, error_medio_relativo_y_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo en y(t)')
    plt.title('Sistema de EDOs 2x2: Comparación de Errores Medios Relativos en y(t)')
    plt.show()

#----------------------------------- Ejecución de las Funciones -----------------------------------
if __name__ == "__main__":
    # Llamamos a cada función para resolver los diferentes casos
    edo_primer_orden()
    edo_segundo_orden()
    sistema_edo_2x2()
