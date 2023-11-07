import tensorflow as tf
from utils import init_model

# PDE T_t + u * grad(T) = alpha * div(grad(T))
def PDE(T, Tt, Tx, Ty, Txx, Tyy):
    return [Tt + 1 * Tx + 1 * Ty - .25 * (Txx + Tyy)]

def get_derivatives(model, X):
    x, y, t = tf.split(X, 3, axis=1)
    with tf.GradientTape(persistent = True) as t2:
        t2.watch(x)
        t2.watch(y)
        with tf.GradientTape(persistent = True) as t1:
            t1.watch(x)
            t1.watch(y)
            t1.watch(t)
            T = model(tf.concat([x, y, t], axis=1))
        Tt = t1.gradient(T, t)
        Tx = t1.gradient(T, x)
        Ty = t1.gradient(T, y)
    Txx = t2.gradient(Tx, x)
    Tyy = t2.gradient(Ty, y)
    return T, Tt, Tx, Ty, Txx, Tyy

# Temperature initial condition
T_A = 1
sx = sy = 1
x0, y0 = -2, -2
def T0(x, y): # T0 = lambda x, y: T_A * tf.exp(- (x**2 / sx**2 + y**2 / sy**2))
    return T_A * tf.exp(- ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))

# Define boundary conditions
def bc1(model, X):
    _, _, _, Ty, _, _ = get_derivatives(model, X)
    return [Ty - 0]

def bc2(model, X):
    _, _, Tx, _, _, _ = get_derivatives(model, X)
    return [Tx - 0]

def bc3(model, X):
    _, _, _, Ty, _, _ = get_derivatives(model, X)
    return [Ty - 0]

def bc4(model, X):
    _, _, Tx, _, _, _ = get_derivatives(model, X)
    return [Tx - 0]

initial_conditions = [T0]
boundary_conditions = [bc1, bc2, bc3, bc4]

model = init_model(input_neurons=3, num_hidden_layers=3, num_neurons_per_layer=50, output_neurons=1, activation='tanh')

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
    'N_iter': 1000,
    'lr': 1e-2,
    'ds': 500,
    'dr': 0.9,
    'patience': 100
}

lbfgs_params = {
    'pi': 8,
    'max_iter': 1000,
}