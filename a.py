import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros del problema
alpha = 0.00012  # Difusividad térmica (m^2/s)
L = 1.0  # Longitud de la barra (m)
T_left = 100  # Temperatura en el extremo izquierdo (°C)
T_right = 100  # Temperatura en el extremo derecho (°C)
T_initial = 20  # Temperatura inicial en toda la barra (°C)
total_time = 1000  # Tiempo total de simulación (s)

# Número de puntos en el espacio
N = 10  # Número de divisiones espaciales
dx = L / (N - 1)  # Tamaño del paso espacial

dt = 0.4 * dx**2 / alpha  # Condición de estabilidad CFL para método explícito
Nt = int(total_time / dt)  # Número de pasos de tiempo

# Parámetro lambda de la ecuación
lambd = alpha * dt / dx**2

# Inicialización de la matriz de temperaturas
T = np.ones((N, Nt)) * T_initial
T[0, :] = T_left  # Condición de frontera izquierda
T[-1, :] = T_right  # Condición de frontera derecha

# Método explícito de diferencias finitas
for j in range(0, Nt - 1):
    for i in range(1, N - 1):
        T[i, j+1] = lambd * T[i+1, j] + (1 - 2 * lambd) * T[i, j] + lambd * T[i-1, j]

# Configuración de la animación
fig, ax = plt.subplots(figsize=(8, 5))
x_vals = np.linspace(0, L, N)
line, = ax.plot(x_vals, T[:, 0], color='red', label=f'N={N}')
ax.set_xlabel("x (m)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Distribución de temperatura en la barra")
ax.set_ylim(0, 110)
ax.legend()
ax.grid()

time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12, color='black')

# Función de actualización para la animación
def update(frame):
    frame = frame % Nt  # Permite que la animación sea un bucle
    line.set_ydata(T[:, frame])
    time_text.set_text(f"Tiempo: {frame * dt:.2f} s")
    ax.set_title(f"Distribución de temperatura en la barra - t={frame * dt:.2f}s")
    return line, time_text

# Crear animación en bucle
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=50, blit=False, repeat=True)

plt.show()