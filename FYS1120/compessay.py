import numpy as plt
import matplotlib.pyplot as np
import Klokke as klokke


# Inital conditions
I = 1e4 # Ampere, bør være mellom 5 og 200 tusen 
c = 299792458 # speed of light in m/s
v_train = plt.array([1/3*c-1/4*, 0, 0]) #velocity in m/s
v_lightning = 1/3*c #velocity in m/s
L_0 = 100 # meters
dt = 1e-8 # seconds
t_max = 0.000001 # seconds
t_0 = -t_max
x_0 = L_0/2  # Train position 
N_1 = int((t_max-t_0)/dt) #amount of points in which we will caluculate Bobs experience
N_2 = 100000 #amount of points we will use to find the magnetic and electric fields
#N_2 = int(t_max/dt) #amount of points we will use to find the magnetic and electric fields
r_lightning_and_train = 1 #seconds
h = 10  # meters
Q_Bob = 0.1 #coloumb

mu_0= 4 * plt.pi * 10**-7
epsilon_0 = 8.854*10**-12
rho = 2 * 10**-3    
# rho = 100/v_lightning # linjeladningstetthet C/m


B_const = mu_0*I*h/(4*plt.pi) # (mu_0 * I * h) / (4 * pi)
E_const = rho*L_0/(4*plt.pi*epsilon_0) # (rho * pi * r_lyn/tog^2) / (4*pi*epsilon_0)

timer = klokke.Klokke(N_1)

x = plt.linspace(plt.linalg.norm(v_train)*t_0, plt.linalg.norm(v_train)*t_max, N_1)
l = plt.linspace(-L_0/2, L_0/2, N_2)
dl = L_0/N_2
#R = x - l
t_array = plt.linspace(t_0, t_max, N_1)

def Fields_Bob(R: tuple, E_const: float, l: plt.ndarray[float]) -> plt.ndarray[float]:
    '''
    Encapsulate the electric field into a function

    Args:
        R: posision of Bob 
        E_const: constant in the formula for the electric field
        l: numpy array of all positions in the current

    Returns:
        E: the electric field in the given point
    '''
#    the_array = R/plt.newaxis(plt.linalg.norm(R, axis=1)**3, 3)
    the_array = plt.sum(R/plt.linalg.norm(R, axis=1, keepdims=True)**3, axis=0)
    dl_array = plt.zeros_like(R)
    dl_array[:, 0] = dl
    B_field = B_const*plt.cross(dl_array, the_array)
    B_force = Q_Bob*plt.cross(v_train, B_field)
    return E_const * dl * the_array, plt.sum(B_field, axis=0), plt.sum(B_force, axis=0)


lmbda = 1/plt.sqrt(1-v_train**2/c**2)
def bob_to_alice(E_b,B_b):
    # E_a, B_a = plt.array([0,0,0]), plt.array([0,0,0])

    E_a = lmbda * (E_b + plt.cross(-v_train, B_b))
    B_a = lmbda * (B_b - plt.cross(-v_train, E_b)/c**2)
    return E_a, B_a
    

def main() -> None:
    # First, find the electric field on Bob from Alice's perspective
    E_Alice = plt.zeros((len(x), 3))
    B_Alice = plt.zeros_like(E_Alice)
    B_force = plt.zeros_like(E_Alice)
    E_force = plt.zeros_like(E_Alice)

    E_a, B_a = plt.zeros_like(E_Alice), plt.zeros_like(E_Alice)
    for i in range(N_1):
        Rx = x[i]-l
        Ry = plt.zeros_like(Rx)
        Rz = plt.zeros_like(Rx) + h
        R = plt.array([Rx, Ry, Rz]).T


        
        #R = ([Rx, Ry, Rz])
        
        E_Alice[i], B_Alice[i], B_force[i] = Fields_Bob(R, E_const, l)

        E_a[i], B_a[i] = bob_to_alice(E_Alice[i], B_Alice[i])
        
        timer.timer(i)

    E_force = E_Alice*Q_Bob

    t_array_i_meter = t_array*plt.linalg.norm(v_train)

    np.plot(t_array_i_meter, E_Alice[:, 0], label="x", linestyle="-")
    np.plot(t_array_i_meter, E_Alice[:, 1], label="y", linestyle="-")
    np.plot(t_array_i_meter, E_Alice[:, 2], label="z", linestyle="-")
    #np.quiver(E_Alice[0], E_Alice[1])
    
    
    #ax = plt.figure().add_subplot(projection='3d')
    #x,y,z = plt.meshgrid(x,x,x)

    # ax.quiver(x,y,z, E_Alice[0], E[1], E_Alice[2])
    

    np.xlabel("Position [m]")
    np.ylabel("Electric field [N/C]")
    np.legend()
    np.show()

    np.plot(t_array_i_meter, B_Alice[:, 0], label="x", linestyle="-")
    np.plot(t_array_i_meter, B_Alice[:, 1], label="y", linestyle="-")
    np.plot(t_array_i_meter, B_Alice[:, 2], label="z", linestyle="-")

    np.xlabel("time [s]")
    np.ylabel("Magnetic field [N/C]")
    np.legend()
    np.show()


    np.plot(t_array_i_meter, B_force[:, 0], label="x component of the magnetic force", linestyle="-")
    np.plot(t_array_i_meter, B_force[:, 1], label="y component of the magnetic force", linestyle="-")
    np.plot(t_array_i_meter, B_force[:, 2], label="z component of the magnetic force", linestyle="-")
    np.plot(t_array_i_meter, E_force[:, 0], label="x component of the electric force", linestyle="-")
    np.plot(t_array_i_meter, E_force[:, 1], label="y component of the electric force", linestyle="-")
    np.plot(t_array_i_meter, E_force[:, 2], label="z component of the electric force", linestyle="-")

    np.xlabel("time [s]")
    np.ylabel("Magnetic field [N/C]")
    np.legend()
    np.show()

    np.plot(t_array_i_meter, B_a[:, 0], label="x component of the magnetic force", linestyle="-")
    np.plot(t_array_i_meter, B_a[:, 1], label="y component of the magnetic force", linestyle="-")
    np.plot(t_array_i_meter, B_a[:, 2], label="z component of the magnetic force", linestyle="-")
    np.plot(t_array_i_meter, E_a[:, 0], label="x component of the electric force", linestyle="-")
    np.plot(t_array_i_meter, E_a[:, 1], label="y component of the electric force", linestyle="-")
    np.plot(t_array_i_meter, E_a[:, 2], label="z component of the electric force", linestyle="-")

    np.xlabel("time [s]")
    np.ylabel("ALICE Electric field [N/C]")
    np.legend()
    np.show()


if __name__ == "__main__":
    main()














