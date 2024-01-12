import tensorflow as tf
from utils import init_model

rho = 1
nu = .5
fx = 0
fy = 0
# Define the PDE
def PDE(u, v, p, ut, vt, ux, vx, uy, vy, px, py, uxx, vxx, uyy, vyy):
    return [
        ux + vy, 
        ut + u * ux + v * uy - nu * (uxx + uyy) + 1 / rho * px - fx, 
        vt + u * vx + v * vy - nu * (vxx + vyy) + 1 / rho * py - fy
    ]

# Get derivatives
def get_derivatives(model, X):
    x, y, t = tf.split(X, 3, axis=1)
    with tf.GradientTape(persistent = True) as t2:
        t2.watch(x)
        t2.watch(y)
        with tf.GradientTape(persistent = True) as t1:
            t1.watch(x)
            t1.watch(y)
            t1.watch(t)
            U = model(tf.concat([x, y, t], axis=1))
            u = U[:, 0:1]
            v = U[:, 1:2]
            p = U[:, 2:3]
        ut = t1.gradient(u, t)
        ux = t1.gradient(u, x)
        uy = t1.gradient(u, y)
        vt = t1.gradient(v, t)
        vx = t1.gradient(v, x)
        vy = t1.gradient(v, y)
        px = t1.gradient(p, x)
        py = t1.gradient(p, y)
    uxx = t2.gradient(ux, x)
    uyy = t2.gradient(uy, y)
    vxx = t2.gradient(vx, x)
    vyy = t2.gradient(vy, y)
    return u, v, p, ut, vt, ux, vx, uy, vy, px, py, uxx, vxx, uyy, vyy

# Define the initial conditions
def u0(x, y):
    return tf.cast((y == 5), tf.float64)

def v0(x, y):
    return 0 * x * y

def p0(x, y):
    return 0 * x * y

# Define boundary conditions
def bc1(model, X):
    U = model(X)
    u, v, _ = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    vars = get_derivatives(model, X)
    py = vars[10]
    return [u - 0, v - 0, py - 0]

def bc2(model, X):
    U = model(X)
    u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    vars = get_derivatives(model, X)
    px = vars[9]
    return [u - 0, v - 0, px - 0]

def bc3(model, X):
    U = model(X)
    u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    vars = get_derivatives(model, X)
    py = vars[10]
    return [u - 1, v - 0, py - 0]

def bc4(model, X):
    U = model(X)
    u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    vars = get_derivatives(model, X)
    px = vars[9]
    return [u - 0, v - 0, px - 0]

initial_conditions = [u0, v0, p0]
boundary_conditions = [bc1, bc2, bc3, bc4]

model = init_model(input_neurons=3, num_hidden_layers=3, num_neurons_per_layer=50, output_neurons=3, activation='tanh')

### Domain ###
x_min, x_max = -5, 5
y_min, y_max = -5, 5
t_min, t_max = 0, 5
domain = {
    'x_min': x_min, 'x_max': x_max,
    'y_min': y_min, 'y_max': y_max,
    't_min': t_min, 't_max': t_max,
}

# Optimizer parameters
adam_params = {
    'N_iter': 30000,
    'lr': 1e-4,
    'ds': 500,
    'dr': 0.9,
    'patience': 100
}

lbfgs_params = {
    'pi': 8,
    'max_iter': 3000,
}