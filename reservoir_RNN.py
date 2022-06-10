import numpy as np
import random
import externally_provided_currents as epc
import scipy as sp
import matplotlib.pyplot as plt
random.seed(2022)
np.random.seed(2022)

gamma = 0.5 # time constant. Larger gamma means RNN responds more strongly/quickly to incoming stimuli u(t)
beta = 0.3 # Tikhonov regularization hyperparameter. Feel free to vary this to find whatever works best.

u_spatial_dim = 3
r_spatial_dim = 100
dt = 0.01

# Generic time stuff
t_start=-1
t_end = 105
times = np.arange(start=t_start,stop=t_end,step=dt)

# Training Time Stuff
t_start_train = 0
t_end_train   = 50
times_training = np.arange(start=t_start_train,stop=t_end_train,step=dt)

# Testing Time Stuff
t_start_test = 50
t_end_test   = 100
times_testing = np.arange(start=t_start_test,stop=t_end_test,step=dt)

W_in = np.random.uniform(low=-1,high=1,size=(r_spatial_dim,u_spatial_dim))
W_out = np.random.uniform(low=-1,high=1,size=(u_spatial_dim,r_spatial_dim))
A = np.random.uniform(low=-1,high=1,size=(r_spatial_dim,r_spatial_dim))
# divide A by spectral radius to help create generalized synchronization
spectral_radius_A = np.max(np.abs(np.linalg.eigvals(A)))
A = A/spectral_radius_A

L63_obj = epc.L63_object()
L63_obj.prepare_f(times)
u = L63_obj
r_initial = np.random.uniform(low=-1,high=1,size=r_spatial_dim)


def calculate_W_out(r, u, beta):
    """
    Calculate linear readout matrix for mapping r(t) back to u(t)
    :param r: (ndarray) dim(space,time)
    :param u: (ndarray) dim(time, space)
    :param beta: (float) Tikhonov regularization parameter
    :return: W_out matrix that maps from r to u.
    """
    # see Lukosevicius Practical ESN eqtn 11
    # https://en.wikipedia.org/wiki/Linear_regression
    # Using ridge regression
    train_start_timestep = 0
    train_end_timestep = r.shape[1]  # length along time axis
    # spatial dimensions
    N_r = sp.shape(r)[0]
    N_u = sp.shape(u)[1]

    # Ridge Regression
    # print("Shape of scipy.matmul(r, r.transpose()): " + str(sp.matmul(r, r.transpose()).shape))
    # print("Shape of scipy.matmul(r, r.transpose()): " + str(sp.identity(N_r).shape))
    W_out = sp.matmul(sp.array(u[train_start_timestep:train_end_timestep, :].transpose()),
                         sp.matmul(r.transpose(),
                                      sp.linalg.inv(
                                          sp.matmul(r, r.transpose()) + beta * sp.identity(N_r))))
    return W_out

def integrate_RNN_driven(r,t,u,gamma,A,W_in):
    drdt = gamma*(-r + np.tanh(A@r + W_in@u.function(N=None,t=t)))
    return drdt

def integrate_RNN_autonomous(r,t,W_out,gamma,A,W_in):
    predicted_u = r@(W_out.T)
    drdt = gamma*(-r + np.tanh(A@r + W_in@predicted_u))
    return drdt

# Driving
RNN_driven_solution_1 = sp.integrate.odeint(integrate_RNN_driven, r_initial, times_training, args=(u,gamma,A,W_in))
# Training
print(RNN_driven_solution_1.shape)
print(u.function(N=None,t=times_training).shape)
W_out = calculate_W_out(r=RNN_driven_solution_1.transpose(),
                        u=u.function(N=None,t=times_training).transpose(),
                        beta=beta)

# Prediction
r_initial_for_testing = RNN_driven_solution_1[-1]#np.random.uniform(low=-1,high=1,size=r_spatial_dim)
RNN_driven_solution_2 = sp.integrate.odeint(integrate_RNN_autonomous, r_initial_for_testing, times_testing, args=(W_out,gamma,A,W_in))

print(W_out.shape)
print(RNN_driven_solution_2.shape)
predicted_u = RNN_driven_solution_2@(W_out.T)

plt.figure()
plt.plot(predicted_u)
plt.title("Gamma="+str(gamma))
plt.show()