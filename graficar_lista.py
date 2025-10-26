#!/usr/bin/env python3
"""
Traza la gráfica de una lista de números decimales usando sus índices como eje X.

Uso:
    python graficar_lista.py 1.2 3.4 2.8 5.0 4.1
"""

import sys
import matplotlib.pyplot as plt

def graficar_lista(valores):
    # Crear lista de índices [0, 1, 2, ...]
    x = list(range(len(valores)))

    # Graficar los puntos y unirlos con una línea
    plt.plot(x, valores, marker='o', linestyle='-', label='Valores ingresados')

    # Etiquetas y estilo
    plt.title("Gráfica de valores")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python graficar_lista.py <valores...>")
        sys.exit(1)

    try:
        # Convertir los argumentos en flotantes
        valores = [float(v) for v in sys.argv[1:]]
    except ValueError:
        print("Error: todos los valores deben ser numéricos.")
        sys.exit(1)

    graficar_lista(valores)
