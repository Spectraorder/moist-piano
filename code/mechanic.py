import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Initial setup
nmax = 5
tnote = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # Onset times of the notes (s)
dnote = np.array([0.4, 0.4, 0.4, 0.4, 0.4])  # Durations of the notes (s)
anote = np.array([0.7, 0.8, 0.9, 0.8, 0.7])  # Relative amplitudes of the notes
inote = np.array([1, 5, 8, 5, 1])

L = 1
J = 81
dx = L / (J - 1)
flow = 220
nstrings = 25

# Additional parameters
moisture = 0.9
temp = 25
pedal_pressed = False

# Calculate temperature effect
temp_effect = 1 - 0.001 * (temp - 20)

# Initialize arrays
f = np.zeros(nstrings)
tau = np.zeros(nstrings)
M = np.ones(nstrings)
T = np.zeros(nstrings)
R = np.zeros(nstrings)
dtmax = np.zeros(nstrings)

for i in range(nstrings):
    f[i] = flow * 2 ** ((i) / 12) * temp_effect
    tau[i] = 1.2 * (440 / f[i]) * (1 - 0.5 * moisture)
    T[i] = M[i] * (2 * L * f[i]) ** 2 * (1 - 0.5 * moisture)
    R[i] = (2 * M[i] * L ** 2) / (tau[i] * np.pi ** 2)
    dtmax[i] = -R[i] / T[i] + np.sqrt((R[i] / T[i]) ** 2 + dx ** 2 / (T[i] / M[i]))

# Determine timestep
tmax_extension = 10 if pedal_pressed else 0
dtmaxmin = np.min(dtmax)
nskip = np.ceil(1 / (8192 * dtmaxmin)).astype(int)
dt = 1 / (8192 * nskip)

tmax = tnote[nmax - 1] + dnote[nmax - 1] + tmax_extension
clockmax = np.ceil(tmax / dt).astype(int)

tstop = np.zeros(nstrings)
H = np.zeros((nstrings, J))
V = np.zeros((nstrings, J))
E = np.zeros(clockmax)  # Initialize energy array

xh1, xh2 = 0.25 * L, 0.35 * L
jstrike = np.arange(np.ceil(1 + xh1 / dx).astype(int), np.floor(1 + xh2 / dx).astype(int))
j = np.arange(1, J - 1)

S = np.zeros((1, np.ceil(clockmax / nskip).astype(int)))[0]
tsave = np.zeros_like(S)

KE = np.zeros(clockmax)  # Kinetic energy
PE = np.zeros(clockmax)  # Potential energy
energy_save = np.zeros_like(S)  # To save energy at each sampled timestep

count = 0
n = 0
pbar = tqdm(total=clockmax, desc='Processing', leave=True)
for clock in range(1, clockmax + 1):
    t = clock * dt
    while (n < nmax) and (tnote[n] <= t):
        V[inote[n], jstrike] = anote[n]
        tstop[inote[n]] = t + dnote[n] + (10 if pedal_pressed else 0)
        n += 1
    for i in range(nstrings):
        if t > tstop[i]:
            H[i, :] = np.zeros(J)
            V[i, :] = np.zeros(J)
        else:
            V[i, j] = V[i, j] + (dt / dx ** 2) * (T[i] / M[i]) * (H[i, j + 1] - 2 * H[i, j] + H[i, j - 1]) + \
                      (dt / dx ** 2) * (R[i] / M[i]) * (V[i, j + 1] - 2 * V[i, j] + V[i, j - 1])
            H[i, j] = H[i, j] + dt * V[i, j]

        # Kinetic energy: 0.5 * mass per unit length * sum of squares of velocity
        KE[clock - 1] += 0.5 * M[i] * dx * np.sum(V[i, :] ** 2)

        # Potential energy: 0.5 * tension * sum of squared differences in displacement divided by dx
        PE[clock - 1] += 0.5 * T[i] * np.sum(np.diff(H[i, :]) ** 2) / dx

    if clock % nskip == 0:
        count += 1
        S[count - 1] = np.sum(H[:, 1])
        tsave[count - 1] = t
        energy_save[count - 1] = KE[clock - 1] + PE[clock - 1]
    pbar.update(1)

pbar.close()

plot_title = f'Simulated Piano Sound Wave - Moisture {moisture}, Temp {temp}°C, Pedal Pressed {pedal_pressed}'

filename = plot_title.replace(' ', '_').replace(',', '').replace('°C', 'C').replace(':', '_').replace('__', '_') + '.png'
filepath = os.path.join('results', filename)

os.makedirs('results', exist_ok=True)

# Plotting Sound Wave
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(tsave[:count], S[:count], label='Sound Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(plot_title)
plt.legend()
plt.grid(True)

# Plotting Energy
plt.subplot(2, 1, 2)
plt.plot(tsave[:count], energy_save[:count], label='Total Energy', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.title('Total Energy of the System Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(filepath)
plt.show()
