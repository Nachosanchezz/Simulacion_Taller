import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros del problema
alpha = 0.001  # Difusividad térmica (m^2/s)
L = 1.0  # Longitud de la barra (m)
T_izquierda = 100  # Condición Dirichlet
T_inicial = 20  # Temperatura inicial

total_time = 300.0  # Tiempo total de simulación (s)
N = 50  # Número de divisiones espaciales
x_vals = np.linspace(0, L, N)
dx = L / (N - 1)

# Cálculo del paso de tiempo
dt = 0.4 * dx**2 / alpha
Nt = int(total_time / dt)
dt = total_time / Nt

# Inicialización de la matriz de temperaturas
T = np.ones((N, Nt)) * T_inicial
T[0, :] = T_izquierda  # Condición Dirichlet en x=0

# lambda
lambd = alpha * dt / dx**2

# Método de diferencias finitas (corregido)
for j in range(0, Nt - 1):
    for i in range(1, N - 1):  # Hasta N-1 para actualizar correctamente todo el dominio excepto extremos
        T[i, j+1] = T[i, j] + lambd * (T[i+1, j] - 2*T[i, j] + T[i-1, j])

    # Condición de Neumann (aislante) en x = L ⇒ T[N-1] = T[N-2]
    T[N-1, j+1] = T[N-2, j+1]

# Configuración de la animación
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x_vals, T[:, 0], color='red', label=f'N={N}')
ax.set_xlabel("x (m)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("Distribución de temperatura en la barra")
ax.set_ylim(0, 110)
ax.legend()
ax.grid()

time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12, color='black')

# Tiempos que queremos mostrar
tiempos = [1, 10, 20, 30, 40, 50, 60, 80, 90, 100, 120, 150, 180, 200, 220, 250, 300]
frames_personalizados = [int(t / dt) for t in tiempos if int(t / dt) < Nt]

# Función de actualización para animación
def update(frame):
    line.set_ydata(T[:, frame])
    tiempo_actual = frame * dt
    time_text.set_text(f"Tiempo: {tiempo_actual:.2f} s")
    ax.set_title(f"Distribución de temperatura - t={tiempo_actual:.2f}s")
    return line, time_text

ani = animation.FuncAnimation(fig, update, frames=frames_personalizados, interval=1000, blit=False, repeat=True)

plt.show()