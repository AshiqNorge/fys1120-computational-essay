import numpy as plt
import matplotlib.pyplot as np

# Inital conditions
I = 5 # Joules
v_train = 1/3
v_lightning = 1/3
L_0 = 1 # seconds
dt = 1e-8 # seconds
t_max = 1 # seconds
x_0 = L_0/2  # Train position 
N_1 = int(t_max/dt) #amount of points in which we will caluculate Bobs experience
N_2 = int(t_max/dt) #amount of points we will use to find the magnetic and electric fields
r_lightning_and_train = 1 #seconds
h = 1e-5  # seconds
B_const = 1 # (mu_0 * I * h) / (4 * pi)
E_const = 1 # (rho * pi * r_lyn/tog^2) / (4*pi*epsilon_0)

x = plt.linspace(0, L_0, N_1)
l = plt.linspace(0, L_0, N_2)
#R = x - l
t_array = plt.linspace(0, t_max, N_1)

def E_0(R: tuple, E_const: float, l: plt.ndarray[float]) -> plt.ndarray[float]:
    '''
    Encapsulate the electric field into a function

    Args:
        R: posision of Bob 
        E_const: constant in the formula for the electric field
        l: numpy array of all positions in the current

    Returns:
        E: the electric field in the given point
    '''

    dl = l[0]
    E_const = E_const*dl    
    the_array = R/plt.abs(R)**3
    
    return E_const * dl * plt.sum(the_array)


def main() -> None:
    # First, find the electric field on Bob from Alice's perspective
    E_Alice = plt.zeros_like(x)




    for i in range(N_1):
        Rx = x[i]-l
        Ry = plt.zeros_like(Rx)
        Rz = plt.zeros_like(Rx) + h
        R = plt.array[Rx, Ry, Rz]
        
        R = ([Rx, Ry, Rz])
        E_Alice[i] = E_0(R, E_const, l)

    np.plot(t_array, E_Alice)
    np.show()


if __name__ == "__main__":
    main()
