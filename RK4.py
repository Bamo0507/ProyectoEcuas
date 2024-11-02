"""
Bryan Martínez 23542
Javier Cifuentes 23079
Adriana Palacios 23044
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math

#---------------------------------------------------------------- Algoritmos -------------------------------------------------------------------
#Método de Runge-Kutta de 4to Orden (RK4)
def runge_kutta_4(f, x0, y0, h, pasos):
    """
    Parámetros a recibir:
    
    f - función descrita por la EDO
    x0 - valor inicial de x
    y0 - valor inicial de y
    h - tamaño de paso (esto afecta la precisión del algoritmo)
    pasos: cantidad de pasos que se darán para encontrar la solución
    """
    
    # Crear arrays para guardar los valores de X y Y
    x_valores = np.zeros(pasos +1) # Array para guardar los valores de X
    y_valores = np.zeros(pasos+1) # Array para guardar los valores de Y
    x_valores[0], y_valores[0] = x0, y0 # Condición inicial aplicada (PVI)
    
    # Ciclo para avanzar desde x0 hasta el último valor que queramos calcular acorde a los pasos definidos
    for i in range(pasos):
        # Calculamos los cuatro "k" que nos ayudarán a estimar el siguiente valor de Y
        # Cada K es una pendiente diferente
        
        # k1 es la pendiente en el punto actual
        k1 = f(x_valores[i], y_valores[i])
        
        # k2 es la pendiente en el punto medio, usando k1 para estimar 
        k2 = f(x_valores[i] + h/2, y_valores[i] + h/2 * k1)
        
        # k3 es otra pendiente en el punto medio, usando k2 para estimar 
        k3 = f(x_valores[i] + h/2, y_valores[i] + h/2 * k2)
        
        # k4 es la pendiente en el punto final, usando k3 para estimar 
        k4 = f(x_valores[i] + h, y_valores[i] + h * k3)
        
        # Ahora, combinamos estas pendientes para obtener el siguiente valor de Y
        y_nuevo = y_valores[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Definimos el siguiente valor de X
        x_valores[i+1] = x_valores[i] + h
        
        # Guardamos el nuevo valor de Y en el array
        y_valores[i+1] = y_nuevo
        
    return x_valores, y_valores

#---------------------------------------------------------------- ------------ -------------------------------------------------------------------



# EDO a resolver PRIMER ORDEN -----------------------------------------------------------------
## Ecuación de primer orden
def f(x, y):
    return (3 * x**3 - y) / (x * math.log(x))
## Solución de la ecuación diferencial
def solucion_analitica(x):
    return (x**3) / np.log(x)
## Parámetros de la simulación
x0 = math.e
y0 = 1
h = 0.1
pasos = 1000


# Solución numérica con Runge-Kutta 4 (RK4)
x_valores_rk4, y_num_rk4 = runge_kutta_4(f, x0, y0, h, pasos)

# Solución analítica en los mismos puntos
y_exacto = solucion_analitica(x_valores_rk4)

# Calcular error absoluto
error_absoluto_rk4 = np.abs(y_exacto - y_num_rk4)

# Mostrar los resultados en la consola
print(" x       | y_num (RK4)      | y_exacto (analítico) | error absoluto")
print("---------------------------------------------------------------------")
for x, y_n, y_e, error in zip(x_valores_rk4, y_num_rk4, y_exacto, error_absoluto_rk4):
    print(f"{x:.2f} | {y_n:.6f}      | {y_e:.6f}        | {error:.6f}")

# Graficar las soluciones para visualizar la comparación
plt.plot(x_valores_rk4, y_num_rk4, label="Runge-Kutta 4 (Numérico)", marker='o')
plt.plot(x_valores_rk4, y_exacto, label="Solución Analítica", linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Comparación entre solución analítica y numérica - EDO Primer Orden (RK4)")
plt.show()
#---------------------------------- -----------------------------------------------------------------
