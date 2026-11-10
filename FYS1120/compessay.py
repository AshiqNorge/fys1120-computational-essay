from platform import system

import numpy as plt
import matplotlib.pyplot as np
import Klokke as klokke



# Inital conditions
I = 1e4                                 # A, bør være mellom 5 og 200 tusen 
c = 299792458                           # m/s, speed of light
v_train = plt.array([1/3*c, 0, 0])  # m/s, speed of the train in Alice's system
v_lightning = 1/3*c                     # m/s, speed of the lightning in Alice's system
gamma = 1 /plt.sqrt(1-(v_train[0]/c)**2) # 1, Lorentz factor

L_0 = 100 *10                               # m, length of lightning channel
h = 10                                  # m, Bobs height above the lightning channel
Q_Henry = 1.6*10**-19                   # C, charge of Henry which Bob holds

dt = 1e-8                               # s, time step
t_max = 1e-5                        # s, maximum time
t_0 = -t_max                            # s, initial time
#x_0 = L_0/2                             # m, initial position of the train
N_1 = int((t_max-t_0)/dt)               # number of points in which we will calculate Bobs experience
N_2 = 100000                             # number of points we will use to find the magnetic and electric fields
dl = L_0 / N_2                          # m, length of each segment of the lightning channel
l = -L_0 / 2+(plt.arange(N_2) + 0.5)*dl  # m, array of positions of the midpoint of each segment of the lightning channel
dl_vec = plt.array([dl, 0.0, 0.0])       # m, vector representing the length of each segment of the lightning channel
t_array = plt.linspace(t_0, t_max, N_1)  # s, time array
x = v_train[0] * t_array                # m, position of Bob at each time step from one end to the other

timer = klokke.Klokke(N_1)
timer2 = klokke.Klokke(N_1)

mu_0= 4 * plt.pi * 10**-7               # H/m, permeability
epsilon_0 = 8.854*10**-12               # F/m, permittivity
rho = 0                                 # C/m, line charge density    (can set to 2 * 10**-3, then we get an E field etc.)

E_const = rho*dl/(4*plt.pi*epsilon_0)  # (rho * pi * r_lyn/tog^2) / (4*pi*epsilon_0)
B_const = mu_0*I/(4*plt.pi)           # (mu_0 * I * h) / (4 * pi)

# timer = klokke.Klokke(N_1)


def fields_at_bob_in_alice(R: plt.ndarray[float]) -> tuple[plt.ndarray[float], plt.ndarray[float]]:
    '''Calculate the electric and magnetic fields at Bob's position in Alice's system.

    Args:
        R: Array with form (N_2, 3). Contains the position vectors from each segment of the lightning channel to Bob's position.

    Returns:
        A tuple containing the electric field and magnetic field at Bob's position in Alice's system.
    '''

    R3 = plt.linalg.norm(R, axis=1, keepdims=True)**3
    R_over_R3 = R / R3

    E = E_const * plt.sum(R_over_R3, axis=0)
    dB = B_const * plt.cross(dl_vec, R_over_R3)
    B = plt.sum(dB, axis=0)

    return E, B

def alice_to_bob(E_Alice: plt.ndarray[float], B_Alice: plt.ndarray[float]) -> tuple[plt.ndarray[float], plt.ndarray[float]]:
    """Transforms the fields at one event from Alice's to Bob's system.
    Args:
        E_Alice: Electric field at Bob's position in Alice's system, with shape (3,)
            and components (Ex, Ey, Ez) in V/m.
        B_Alice: Magnetic field at Bob's position in Alice's system, with shape (3,)
            and components (Bx, By, Bz) in T.

    Returns:
        A pair (E_Bob, B_Bob) with the fields at the same event expressed
        in Bob's system.
    """
    E_Bob = gamma * (E_Alice + plt.cross(v_train, B_Alice))
    B_Bob = gamma * (B_Alice - plt.cross(v_train, E_Alice) / c**2)

    # Component parallel to the movement direction is continuous.
    E_Bob[0] = E_Alice[0]
    B_Bob[0] = B_Alice[0]

    return E_Bob, B_Bob


def main() -> None:
    """Calculate the electric and magnetic fields at Bob's position in both Alice's and Bob's systems, and plot the results."""
    # First, find the electric field on Bob from Alice's perspective
    E_Alice = plt.zeros((N_1, 3))
    B_Alice = plt.zeros_like(E_Alice)
    E_Bob = plt.zeros_like(E_Alice)
    B_Bob = plt.zeros_like(E_Alice)

    R = plt.zeros((N_2, 3))     # Empty position array
    R[:, 2] = h                 # Set the z-component to Bob's height


    # Calculate the fields at Bob's position for each time step.
    # for i, x_Bob in enumerate(x):
    #     R[:, 0] = x_Bob - l     # Update the R-vector as Bob has moved
    #     E_Alice[i], B_Alice[i] = fields_at_bob_in_alice(R)
    #     E_Bob[i], B_Bob[i] = alice_to_bob(E_Alice[i], B_Alice[i])
    #     timer.timer(i)


    #calculates Henry stuff --------------
    R_Henry = plt.copy(R)
    a_Henry = plt.zeros(len(x))
    v_Henry = plt.zeros(len(x))
    t_Henry = 0
    fail = True
    Henry_position = plt.zeros((len(x), 3))
    Henry_position[:, 0] = x
    Henry_position[:, 2] = h
    Henry_timearray = plt.zeros(len(x))
    Henry_timearray += t_0
    proton_mass = 1.6e-27 #kg
    
    a_Henry_Bob = plt.copy(a_Henry)
    v_Henry_Bob = plt.copy(v_Henry)
    x_Henry_Bob = plt.copy(Henry_position)
    x_Henry_Bob[:, 0] = x
    x_Henry_Bob[:, 2] = h

    #Bruker LeapFrog for å integrere Henrys bane
    for i, x_Bob in enumerate(x):
        R_Henry[:, 0] = x_Bob-l
        Henry_position[i, 2] = Henry_position[i-1, 2] + 0.5*a_Henry[i-1]*dt**2 + v_Henry[i-1]*dt
        R_Henry[:, 2] = Henry_position[i, 2]
        field_elec, field_magn = fields_at_bob_in_alice(R_Henry) #finds Henry's acceleration in Alices system
        a_Henry[i] = field_elec[2]*Q_Henry/proton_mass + (plt.cross(plt.array([v_train[0], 0, v_Henry[i-1]]), field_magn)[2]*Q_Henry/proton_mass)
        v_Henry[i] = v_Henry[i-1] + 0.5*(a_Henry[i-1]+a_Henry[i])*dt
        t_Henry += dt
        Henry_timearray[i] += t_Henry

        #In Bobs frame of referance:
        x_Henry_Bob[i, 2] = x_Henry_Bob[i-1, 2] + 0.5*a_Henry_Bob[i-1]*dt**2 + v_Henry_Bob[i-1]*dt
        field_elec_Bob, field_magn_Bob = alice_to_bob(field_elec, field_magn)
        a_Henry_Bob[i] = field_elec[2]*Q_Henry/proton_mass + (plt.cross(plt.array([0, 0, v_Henry[i-1]]), field_magn_Bob)[2]*Q_Henry/proton_mass)
        v_Henry_Bob[i] = v_Henry_Bob[i-1] + 0.5*(a_Henry[i-1] + a_Henry[i])*dt
        if Henry_position[i, 2] <= 0:
            print(f"{Henry_position[i]=}, {t_Henry=}, {x_Bob=}")
            N_Henry = i+1
            fail = False
            break
        timer2.timer(i)
    if fail == True:
        print(f"Henrykoden rakk ikke kjøre, øk t_max for bedre resultater. Henrys posisjon er nå: {Henry_position[-1]}")

    print(f"{Henry_position[0]=}, {Henry_timearray[0]=}")
    #Problem: Posisjonen på x-aksen vi får fra koden er åpenbart feil. Vi får -220m, men vi skal få 780m


    np.subplot(1, 2, 1)
    np.plot(Henry_timearray[:N_Henry], x_Henry_Bob[:N_Henry, 2], "o")
    np.xlabel("Time [s]")
    np.ylabel("position [m]")
    np.title("Bob ref.syst.")

    np.subplot(1, 2, 2)
    np.plot(Henry_timearray[:N_Henry], Henry_position[:N_Henry, 2], "o")
    np.xlabel("Time [s]")
    np.ylabel("position [m]")
    np.title("Alice ref.syst.")
    np.show()




    '''
    # Calculate the forces on Bob in both systems.
    F_electric_Alice = Q_Henry * E_Alice
    F_magnetic_Alice = Q_Henry * plt.cross(v_train, B_Alice)
    F_Alice = F_electric_Alice + F_magnetic_Alice

    F_Bob = Q_Henry * E_Bob   # Bob does see a magnetic field, but it does not exert a force on him since he is at rest in his own system.

    # fig is a matplotlib figure object, and ax is a 2x2 array of subplots, with the first index being the row and the second index being the column.
    fig, ax = np.subplots(2, 2, layout="constrained")

    ax[0, 0].plot(x, E_Alice[:, 0], label="Alice")
    ax[0, 0].plot(x, E_Bob[:, 0], "--", label="Bob")
    ax[0, 0].set_ylabel("Ex [V/m]")

    ax[0, 1].plot(x, E_Alice[:, 2], label="Alice")
    ax[0, 1].plot(x, E_Bob[:, 2], "--", label="Bob")
    ax[0, 1].set_ylabel("Ez [V/m]")

    ax[1, 0].plot(x, B_Alice[:, 1], label="Alice")
    ax[1, 0].plot(x, B_Bob[:, 1], "--", label="Bob")
    ax[1, 0].set_ylabel("By [T]")

    ax[1, 1].plot(x, F_electric_Alice[:, 2], ":", label="Alice, electric force")
    ax[1, 1].plot(x, F_magnetic_Alice[:, 2], ":", label="Alice, magnetic force")
    ax[1, 1].plot(x, F_Alice[:, 2], label="Alice, total force")
    ax[1, 1].plot(x, F_Bob[:, 2], "--", label="Bob, total force")
    ax[1, 1].set_ylabel("Force in z direction [N]")

    for subplot in ax.flat:
        subplot.legend()        # Turns on the legend for each subplot
        subplot.grid(alpha=0.3) # Turns on gridnet with 30% transparency

    fig.supxlabel("Bobs position in Alice's system [m]")
    np.show()
    '''

if __name__ == "__main__":
    main()
    




'''
Kort diskusjon. Si at når Bob passerer Alice, så dropper han en positiv ladning rett ned.
Begge vil se at den får en kraft nedover, men vil være uenig om kraftn:
- Alice ser at ladningen beveger seg med toget, og forklarer kraft nedover som magnetkraft (qv x B) fra lynkanalen.
- Bob ser at ladningen er i ro, og forklarer kraft nedover med sitt elektriske felt (qE_z) som kommer fra lynkanalen

Da kan vi forklare hvordan Alice kun ser B felt, mens Bob ser E og B felt.
Alice: Vi definerer ledningen som null netto ladning, da får vi kun B felt fra Bob's bevegelse
Bob: Han ser på en kanaldel til høyre for han. Kanaldelen inneholder positive ioner som står stille, og negative ioner som beveger seg til venstre. Han ser på tettheten til ladningene, fra Alice's system så registerer Bob det høyre elektronet litt senere enn venstre side. I denne tiden har elektronene beveget seg mot venstre, og avstanden Bob måler mellom dem er mindre enn ved de positive ionene. Bob ser derfor en negativ netto ladning.
'''








