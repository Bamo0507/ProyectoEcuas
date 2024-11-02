"""
Bryan Martínez 23542
Javier Cifuentes 23079
Adriana Palacios 23044
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math

#---------------------------------------------------------------- Algoritmos -------------------------------------------------------------------
#Método de Adams Moulton
def adams_moulton(f, x0, y0, h, pasos):
    """
    Párametros a recibir:
    
    f - función descrita por la EDO
    x0 - valor inicial de x
    y0 - valor inicial de y
    h - tamaño de paso (esto afecta la precisión del algoritmo)
    pasos: cantidad de pasos que se darán para encontrar la solución
    
    """
    x_valores = np.zeros(pasos +1) #se crea un array para guardar los valores de X acorde a los steps que usemos
    y_valores = np.zeros(pasos+1) #se crea un array para guardar los valores de Y acorde a los steps que usemos
    x_valores[0], y_valores[0] = x0, y0 #condición inicial aplicada (PVI)
    
    #Ciclo para avanzar desde x sub 0 hasta el último valor que queramos calcular acorde a los pasos definidos
    for i in range(pasos):
        
        #predicción inicial para el valor de y sub n + 1
        #y_valores[i] - valor actual que llevamos en Y
        #f(x_valroes[i], y_valroes[i]) - derivada en el punto actual
        #h - paso de qué tanto avanzamos x iteración
        y_predic = y_valores[i] + h * f(x_valores[i], y_valores[i])
        
        
        
        #Definimos el siguietne valor de x a utilizar
        #Para hacerlo le agregamos nuestro h (tamaño de paso) al actual valor de x
        x_valores[i+1] = x_valores[i] + h
        
        
        
        #Se hace una corrección sobre el valor que se predijo
        #El valor predijo es el que se obtuvo al evaluar la derivada
        #Lo que se hace es una media de pendientes para ajustar y_predij
        #El promedio lo hace entre la pendiente en el punto inicial y en el punto final
        y_nuevo = y_valores[i] + h/2 * (f(x_valores[i], y_valores[i]) + f(x_valores[i+1], y_predic))
        
        
        #Parámetros para el proceso de refinamiento del valor de y sub n + 1
        #parará el ciclo hasta que la diferencia entre el y actual y el pasado sea menor al valor de la variable tol
        tol = 1e-6 
        
        #Intentos máximos a hacer para corregir el valor de y, este es el que manda
        #Aunque no se cumpla que sea menor que tol, si pasaron las 10 iteraciones ahí llegamos
        max_iter = 10 
        
        #Variable de conteos para llevar cuántas iteraciones van y así detener el while si amerita
        iter_count = 0
        
        while abs(y_nuevo - y_predic) > tol and iter_count < max_iter:
            y_predic = y_nuevo
            #Igual que antes e van haciendo promedios con el punto inicial y final haciendolo más "exacto"
            y_nuevo = y_valores[i] + h/2 * (f(x_valores[i], y_valores[i]) + f(x_valores[i+1], y_predic))
            iter_count += 1 #sube la cuenta de iteraciones para parar
        
        #Se almacena el valor corregido y_[n+1]
        y_valores[i+1] = y_nuevo
        
    return x_valores, y_valores

#---------------------------------------------------------------- ------------ -------------------------------------------------------------------



#EDO a resolver PRIMER ORDEN -----------------------------------------------------------------
##Ecuación de primer orden
def f(x, y):
    return (3 * x**3 - y) / (x * math.log(x))
##Solución de la ecuación diferencial
def solucion_analitica(x):
    return (x**3) / np.log(x)
## Parámetros de la simulación
x0 = math.e
y0 = 1
h = 0.01
pasos = 1000


# Solución numérica con Adams-Moulton
x_valores, y_num = adams_moulton(f, x0, y0, h, pasos)

# Solución analítica en los mismos puntos
y_exacto = solucion_analitica(x_valores)

# Calcular error absoluto
error_absoluto = np.abs(y_exacto - y_num)

# Mostrar los resultados en la consola
print(" x       | y_num (numérico) | y_exacto (analítico) | error absoluto")
print("---------------------------------------------------------------")
for x, y_n, y_e, error in zip(x_valores, y_num, y_exacto, error_absoluto):
    print(f"{x:.2f} | {y_n:.6f}      | {y_e:.6f}        | {error:.6f}")

# Graficar las soluciones para visualizar la comparación
plt.plot(x_valores, y_num, label="Adams-Moulton (Numérico)", marker='o')
plt.plot(x_valores, y_exacto, label="Solución Analítica", linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Comparación entre solución analítica y numérica - ED Primer Orden")
plt.show()
#---------------------------------- -----------------------------------------------------------------


