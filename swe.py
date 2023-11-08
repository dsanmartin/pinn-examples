import tensorflow as tf
from utils import init_model

H = 0
f = 1
g = 9.8
k = 1
nu = 1
# Define the PDE
def PDE(h, u, v, ht, ut, vt, hx, ux, vx, hy, uy, vy, uxx, vxx, uyy, vyy):
    return [
        ht + hx * u + h * ux + hy * v + h * vy, 
        ut + u * ux + v * uy - f * v + g * hx + k * u - nu * (uxx + uyy), 
        vt + u * vx + v * vy + f * u + g * hy + k * v - nu * (vxx + vyy)
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
            h = U[:, 0:1]
            u = U[:, 1:2]
            v = U[:, 2:3]
        ht = t1.gradient(h, t)
        hx = t1.gradient(h, x)
        hy = t1.gradient(h, y)
        ut = t1.gradient(u, t)
        ux = t1.gradient(u, x)
        uy = t1.gradient(u, y)
        vt = t1.gradient(v, t)
        vx = t1.gradient(v, x)
        vy = t1.gradient(v, y)
    uxx = t2.gradient(ux, x)
    uyy = t2.gradient(uy, y)
    vxx = t2.gradient(vx, x)
    vyy = t2.gradient(vy, y)
    return h, u, v, ht, ut, vt, hx, ux, vx, hy, uy, vy, uxx, vxx, uyy, vyy

# Define the initial conditions
h_A = 1
sx = sy = 1
x0, y0 = 0, 0
def h0(x, y):
    return h_A * tf.exp(- ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))

def u0(x, y):
    return 0 * x * y

def v0(x, y):
    return 0 * x * y


# Define boundary conditions
def bc1(model, X):
    h, u, v, ht, ut, vt, hx, ux, vx, hy, uy, vy, uxx, vxx, uyy, vyy = get_derivatives(model, X)
    return [hy - 0, uy - 0, vy - 0]

def bc2(model, X):
    h, u, v, ht, ut, vt, hx, ux, vx, hy, uy, vy, uxx, vxx, uyy, vyy = get_derivatives(model, X)
    return [hx - 0, ux - 0, vx - 0]

def bc3(model, X):
    h, u, v, ht, ut, vt, hx, ux, vx, hy, uy, vy, uxx, vxx, uyy, vyy = get_derivatives(model, X)
    return [hy - 0, uy - 0, vy - 0]

def bc4(model, X):
    h, u, v, ht, ut, vt, hx, ux, vx, hy, uy, vy, uxx, vxx, uyy, vyy = get_derivatives(model, X)
    return [hx - 0, ux - 0, vx - 0]

initial_conditions = [h0, u0, v0]
boundary_conditions = [bc1, bc2, bc3, bc4]

model = init_model(input_neurons=3, num_hidden_layers=3, num_neurons_per_layer=50, output_neurons=3, activation='tanh')

### Domain ###
x_min, x_max = -5, 5
y_min, y_max = -5, 5
t_min, t_max = 0, 1
domain = {
    'x_min': x_min, 'x_max': x_max,
    'y_min': y_min, 'y_max': y_max,
    't_min': t_min, 't_max': t_max,
}

# Optimizer parameters
adam_params = {
    'N_iter': 5000,
    'lr': 1e-2,
    'ds': 100,
    'dr': 0.9,
    'patience': 100
}

lbfgs_params = {
    'pi': 8,
    'max_iter': 1000,
}