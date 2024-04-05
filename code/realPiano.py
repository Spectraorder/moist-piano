import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


def smooth_adsr_envelope(t, attack=0.01, decay=0.3, sustain_level=0.6, release=0.5, duration=2.0):
    """Generate a smoother ADSR envelope."""
    if t < attack:
        # Smooth transition from 0 to 1
        return np.power(t / attack, 2)
    elif t < attack + decay:
        # Smooth transition from 1 to sustain level
        return 1 - (1 - sustain_level) * np.power((t - attack) / decay, 2)
    elif t < duration - release:
        return sustain_level
    elif t < duration:
        # Smooth transition from sustain level to 0
        return sustain_level * (1 - np.power((t - (duration - release)) / release, 2))
    else:
        return 0


# Simulation parameters
sampling_rate = 44100
duration = 2.0
t = np.linspace(0, duration, int(sampling_rate * duration))

# Harmonic synthesis
fundamental_freq = 440
harmonics = [1, 2, 3, 4, 5]
amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625]
sound = np.zeros_like(t)

for i, harmonic in enumerate(harmonics):
    frequency = fundamental_freq * harmonic
    amplitude = amplitudes[i]
    sound += amplitude * np.sin(2 * np.pi * frequency * t)

envelope = np.array(
    [smooth_adsr_envelope(ti, attack=0.01, decay=0.3, sustain_level=0.6, release=0.5, duration=duration) for ti in t])
sound *= envelope

sound = sound / np.max(np.abs(sound))

write("piano_simulation.wav", sampling_rate, sound.astype(np.float32))

plt.figure(figsize=(10, 4))
plt.plot(t, sound)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Simulated Piano Sound Wave with Smooth ADSR Envelope')
plt.show()
