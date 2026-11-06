import numpy as np
import matplotlib.pyplot as plt

c = 299_792_458.0             # m/s
mu0 = 4 * np.pi * 1e-7        # H/m, tilstrekkelig presisjon her
eps0 = 1 / (mu0 * c**2)
L, I = 100.0, 5.0             # m og A, valgte eksempelverdier
h0 = 10.0                     # referansehøyde i meter
v = c / 3
linjeladning = 0.0            # netto C/m i Alices system

def felt_i_alices_system(x, h, ladning, n=400):
    # Returner Ex, Ez og By ved (x, 0, h), alle i SI-enheter.
    ds = L / n
    s = -L / 2 + (np.arange(n) + 0.5) * ds
    rx = x - s
    r3 = (rx**2 + h**2)**1.5
    k = ladning / (4 * np.pi * eps0)
    ex = k * np.sum(rx / r3) * ds
    ez = k * np.sum(h / r3) * ds
    by = -mu0 * I / (4 * np.pi) * np.sum(h / r3) * ds
    return ex, ez, by


def til_bob(ex, ez, by, fart):
    gamma = 1 / np.sqrt(1 - (fart / c)**2)
    return ex, gamma * (ez + fart * by), gamma * (by + fart * ez / c**2)

ex, ez, by = felt_i_alices_system(0, h0, linjeladning)
ex_b, ez_b, by_b = til_bob(ex, ez, by, v)
print(f"Alice: Ez = {ez:.2f} V/m, By = {by * 1e6:.4f} µT")
print(f"Bob:   Ez = {ez_b:.2f} V/m, By = {by_b * 1e6:.4f} µT")

# G = L / (h0 * np.sqrt((L / 2)**2 + h0**2))

# by_eksakt = -mu0 * I * G / (4 * np.pi)
# for n in [20, 40, 80, 400]:
#     by_num = felt_i_alices_system(0, h0, linjeladning, n)[2]
#     feil = abs((by_num - by_eksakt) / by_eksakt)
#     print(f"{n:3d} biter: relativt avvik i By = {feil:.2e}")