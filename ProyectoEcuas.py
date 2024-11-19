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
    El método numérico de Adams Moulton es un poco complejo, pues es
    de tipo predictor-corrector, esto implica que primero va a hacer
    una estimación del siguiente valor, y luego le realiza una serie de 
    correcciones para dejar la respuesta más precisa.
    
    Para realizar estas "predicciones" se utiliza el método de Euler:
    ypred = yn + h*f(xn, yn),
    aquí, el yn es el valor actual que se ha calculado, el h es el 
    tamaño del paso, mientras más pequeño más preciso es, y 
    el f(xn, yn) representa la derivada sobre el punto actual. 
    
    Además, la manera en que funcionan las correcciones es que se calcula
    la pendiente promedio entre el punto en el que estamos ahorita, contra 
    el punto que se haya predicho.
    
    En los parámetros del método lo que se manda es:
    - funcion: la ecuación diferencial despejada para la y de mayor orden.
    - x0: valor inicial en x.
    - y0: valor inicial en y.
    - h: tamaño del paso.
    - pasos: la cantidad de pasos que se van a dar, mientras más, mejor 
    precisión se tendrá.
    """
    
    """
    Se verifica si lo que se mandó como parámetro es un número,
    esto para poder manejar los caso sen donde se deje de trabajar
    con ecuaciones diferenciales de primer y segundo orden, y ya
    estemos con el sistema de EDs 2x2.
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
        # La tolerancia establece que se espera que se deje de 
        # corregir hasta que la diferencia entre el valor nuevo
        # y el pasado sea tan pequeño como ese valor.
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
    El método númerico de cuarto orden funciona considerablemente
    diferente al de adams-moulton, pues ahora ya no estamos prediciendo
    y corrigiendo valores, si no que calculamos 4 pendientes para su 
    funcionamiento. La primera pendiente se hace en el punto inicial, 
    luego, bajo un intervalo, trabajamos la segunda y tercera pendiente
    en un punto medio, y la cuarta pendiente se saca del final del intervalo.
    
    Al finalizar este cálculo, se pasa a hacer una combinación de las 
    pendientes encontradas para poder encontrar el próximo valor de y.
    
    La manera en que se combinan es haciendo una combinación ponderada entre
    las cuatro pendientes, la fórmula que se utiliza es:
    
    yn+1 = yn + (h/6)(k1+2k2+2k3+k4)
    
    Una vez más, lo que se manda en los parámetros es:
    - funcion: la ED despejada para la y de mayor orden. 
    - x0: valor inicial de x.
    - y0: valor inicial de y.
    - h: tamaño de paso.
    - pasos: cantidad de pasos
    """
    
    """
    Se verifica si lo que se mandó como parámetro es un número,
    esto para poder manejar los caso sen donde se deje de trabajar
    con ecuaciones diferenciales de primer y segundo orden, y ya
    estemos con el sistema de EDs 2x2.
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
    # Ecuación de primer orden: (xlnx)y'+y=3x^3
    # PVI: y(e) = 1
    # Solución analítica: y = (X^3+(1-e^3))/(lnx)
    
    """
    Se establece la funcón a utilizar, aquí ya se despejó
    la ED para el y'.
    """
    def funcion(x, y):
        return (3 * x**3 - y[0]) / (x * np.log(x))
    
    """
    Se define la solución analítica que se encontró, para luego 
    ir ploteando y contrastar con la gráfica que generen los
    2 métodos númericos.
    """
    def solucion_analitica(x):
        return (x**3 + (1 - np.exp(3)))/ np.log(x)
    
    # PVIs
    x0 = math.e
    y0 = 1
    
    # Tamaño de paso y cantidad que se darán
    h = 0.01
    pasos = 10000

    """
    Se recogen los resultados para los puntos en 'x' y en 'y'
    para ambos métodos núericos para luego graficar
    """
    x_vals_am, y_vals_am = adams_moulton(funcion, x0, y0, h, pasos)
    x_vals_rk4, y_vals_rk4 = runge_kutta_4(funcion, x0, y0, h, pasos)
    y_exacta = solucion_analitica(x_vals_am)

    # Calcular el error relativo
    """
    Esto para saber de cuánto fue el porcentaje de rror presentado por cada método
    básicamente lo que se hace es que agarramos todos los puntos cálculados,
    les restamos el valor según la solución analítica en ese punto 'x', y lo 
    dividimos por el valor dado por la analítica, así logramos saber qué tan certero
    fue. Esto tras hacerlo para cada punto, le sacamos el promedio de error.
    """
    error_relativo_am = np.abs((y_vals_am[:, 0] - y_exacta) / y_exacta)
    error_relativo_rk4 = np.abs((y_vals_rk4[:, 0] - y_exacta) / y_exacta)
    error_medio_relativo_am = np.mean(error_relativo_am)
    error_medio_relativo_rk4 = np.mean(error_relativo_rk4)
    print("ED de Primer Orden:")
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
    plt.title("ED de Primer Orden")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_am, error_medio_relativo_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo')
    plt.title('ED de Primer Orden')
    plt.show()

#----------------------------------- EDO de Segundo Orden -----------------------------------
def edo_segundo_orden():
    # Ecuación de segundo orden: y'' - 2y' + y = x e^x + 4
    # PVIs: y(0) = 1, y'(0) = 1
    # Solución analítica: y = (1/6)x^3*e^x + 4xe^x -3e^x + 4

    """
    Se establece la función a utilizar, se hacen algunos cambios
    pues para trabajar con EDs de segundo orden, si bien despejamos 
    siempre para la y'', luego debemos realizar pequeñas sustituciones
    para lograr manejar el y' y la y, para hacerlo se planetea un mini 
    sistema de ecuaciones, en donde la y se representa con y1, y la y'
    como y2.
    """
    def funcion(x, y):
        y1 = y[0]
        y2 = y[1]
        dy1_dx = y2
        dy2_dx = 2 * y2 - y1 + x * np.exp(x) + 4
        return np.array([dy1_dx, dy2_dx])
    
    """
    Se da la función a la que se llegó como solución a la ED de segundo orden.
    """
    def solucion_analitica(x):
        return (1/6)*x**3 * np.exp(x) + 4*x * np.exp(x) -3* np.exp(x) + 4
    
    # PVIs
    x0 = 0
    y0 = np.array([1, 1])  # y(0) = 1, y'(0) = 1
    
    # Tamaño y cnatidad de pasos
    h = 0.01
    pasos = 10000

    """
    Se recogen los resultados para los puntos en 'x' y en 'y'
    para ambos métodos núericos para luego graficar
    """
    x_vals_am, y_vals_am = adams_moulton(funcion, x0, y0, h, pasos)
    x_vals_rk4, y_vals_rk4 = runge_kutta_4(funcion, x0, y0, h, pasos)
    y_exacta = solucion_analitica(x_vals_am)

    """
    Como ya se mencionó en esta sección de la ED de primer orden, 
    para lograr contrastar y saber qué tan precisos son los métodos, y
    cómo se compara uno con el otro, se calcula el error relativo en cada
    punto para cada método, contrastando lo que se obtuvo en la solución
    analítica, y luego promediando el error.
    """
    error_relativo_am = np.abs((y_vals_am[:, 0] - y_exacta) / y_exacta)
    error_relativo_rk4 = np.abs((y_vals_rk4[:, 0] - y_exacta) / y_exacta)
    error_medio_relativo_am = np.mean(error_relativo_am)
    error_medio_relativo_rk4 = np.mean(error_relativo_rk4)
    print("\nED de Segundo Orden:")
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
    plt.title("ED de Segundo Orden")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_am, error_medio_relativo_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo')
    plt.title("ED de Segundo Orden")
    plt.show()

#----------------------------------- Sistema de EDOs 2x2 -----------------------------------
def sistema_edo_2x2():
    # Sistema de ecuaciones diferenciales:
    # x' = 4x + 5y
    # y' = -2x + 6y
    # PVI: x(0) = 1, y(0) = 1
    
    """
    Se establece la función a utilizar para el sistema de ecuaciones diferenciales.
    En este caso, ya contamos con un sistema de EDOs de primer orden, por lo que
    no es necesario realizar sustituciones adicionales. Simplemente definimos cada
    ecuación del sistema utilizando las variables correspondientes. Asignamos las
    derivadas x' y y' a las expresiones dadas por el sistema, y proporcionamos
    las condiciones iniciales para cada variable.
    """
    def funcion(t, y):
        x = y[0]
        y_var = y[1]
        dx_dt = 4 * x + 5 * y_var
        dy_dt = -2 * x + 6 * y_var
        return np.array([dx_dt, dy_dt])
    
    """
    Proporcionamos la solución a la que se llego por el medio analítico, tomando
    lo que corresponde para x y lo que es para y.
    """
    def solucion_analitica(t):
        x_exacta = (1/5)*np.exp(5*t)*(5*np.cos(3*t)) + (4/15)*np.exp(5*t)*(5*np.sin(3*t))
        y_exacta = (1/5)*np.exp(5*t)*(np.cos(3*t) - 3*np.sin(3*t)) + (4/15)*np.exp(5*t)*(3*np.cos(3*t) + np.sin(3*t))
        return x_exacta, y_exacta
    
    #PVIs
    t0 = 0
    y0 = np.array([1, 1])
    
    #Tamaño y cantidad de pasos
    h = 0.01
    pasos = 10000

    """
    Cálculamos los puntos para nuestra solución analítica, y aplicando
    los dos métodos numéricos.
    """
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
    print("\nSistema de EDs 2x2")
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
    plt.title("Sistema de EDs 2x2 Soluciones x(t)")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras para x(t)
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_x_am, error_medio_relativo_x_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo en x(t)')
    plt.title('Sistema de EDs 2x2 Error Medio Relativo x(t)')
    plt.show()
    
    # Graficar las soluciones para y(t)
    plt.figure(figsize=(12, 6))
    plt.plot(t_vals_am, y_vals_am[:, 1], label="Adams-Moulton y(t)", linestyle='-', color='blue')
    plt.plot(t_vals_rk4, y_vals_rk4[:, 1], label="RK4 y(t)", linestyle='--', color='green')
    plt.plot(t_vals_am, y_exacta_am, label="Solución Analítica y(t)", linestyle=':', color='red')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.title("Sistema de EDs 2x2 Soluciones y(t)")
    plt.grid(True)
    plt.show()

    # Graficar el error medio relativo en un gráfico de barras para y(t)
    plt.figure()
    metodos = ['Adams-Moulton', 'RK4']
    errores = [error_medio_relativo_y_am, error_medio_relativo_y_rk4]
    plt.bar(metodos, errores, color=['blue', 'green'])
    plt.ylabel('Error Medio Relativo en y(t)')
    plt.title('Sistema de EDs 2x2 Error Medio Relativo y(t)')
    plt.show()

#----------------------------------- Ejecución de las Funciones -----------------------------------
if __name__ == "__main__":
    # Llamamos a cada función para resolver los diferentes casos
    edo_primer_orden()
    edo_segundo_orden()
    sistema_edo_2x2()
