import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros del problema
alpha = 0.001  # Difusividad térmica (m^2/s)
L = 1.0  # Longitud de la barra (m)
T_left = 0  # Temperatura en el extremo izquierdo (°C)
T_right = 0  # Temperatura en el extremo derecho (°C)
total_time = 50.0  # Tiempo total de simulación (s)

# Número de puntos en el espacio
N = 50  # Número de divisiones espaciales
x_vals = np.linspace(0, L, N)
dx = L / (N - 1)  # Tamaño del paso espacial

# Condición inicial: T(x,0) = 10 * sin(pi * x)
dt = 0.4 * dx**2 / alpha
Nt = int(total_time / dt)
dt = total_time / Nt  # Ajuste exacto al tiempo final

# Inicialización de la matriz de temperaturas
T = np.zeros((N, Nt))
T[:, 0] = 10 * np.sin(np.pi * x_vals)
T[0, :] = T_left  # Condición de frontera izquierda
T[-1, :] = T_right  # Condición de frontera derecha

# Parámetro lambda de la ecuación
lambd = alpha * dt / dx**2

# Método explícito de diferencias finitas
for j in range(0, Nt - 1):
    for i in range(1, N - 1):
        T[i, j+1] = T[i, j] + lambd * (T[i+1, j] - 2*T[i, j] + T[i-1, j])

# Configuración de la animación
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x_vals, T[:, 0], color='red', label=f'N={N}')
ax.set_xlabel("x (m)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Distribución de temperatura en la barra")
ax.set_ylim(0, 11)
ax.legend()
ax.grid()

time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12, color='black')

# Selección de frames para animación: 1s, 10s, 20s, 30s, 40s, 50s
tiempo_objetivo = [1, 10, 20, 30, 40, 50]
frames_personalizados = [int(t / dt) for t in tiempo_objetivo if int(t / dt) < Nt]

# Función de actualización para la animación
def update(frame):
    line.set_ydata(T[:, frame])
    tiempo_actual = frame * dt
    time_text.set_text(f"Tiempo: {tiempo_actual:.2f} s")
    ax.set_title(f"Distribución de temperatura en la barra - t={tiempo_actual:.2f}s")
    return line, time_text

# Crear animación en bucle solo con los tiempos deseados
ani = animation.FuncAnimation(fig, update, frames=frames_personalizados, interval=1000, blit=False, repeat=True)

plt.show()
