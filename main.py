import os
import sys
import time
import numpy as np
import tensorflow as tf
from plots import loss_plot

DTYPE = 'float64'
NUM_THREADS = 4
MODEL_DIR = 'models'
LOG_OUT = 'i: {:08d}: Total loss = {:2.4e}, PDE loss = {:2.4e}, BC loss = {:2.4e}, IC loss = {:2.4e}'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
# os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
# os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)
# tf.config.threading.set_inter_op_parallelism_threads(
#     num_threads
# )
# tf.config.threading.set_intra_op_parallelism_threads(
#     num_threads
# )
# tf.config.set_soft_device_placement(True)
tf.keras.backend.set_floatx(DTYPE)

# Model name using date and time
model_name = time.strftime('%Y%m%d%H%M%S')
load_model_dir = sys.argv[1] if len(sys.argv) > 1 else None

# Temperature initial condition
T_A = 1
sx = sy = 1
def T0(x, y): # T0 = lambda x, y: T_A * tf.exp(- (x**2 / sx**2 + y**2 / sy**2))
    return T_A * tf.exp(- (x**2 / sx**2 + y**2 / sy**2))

# MLP definition
def init_model(num_hidden_layers=3, num_neurons_per_layer=50, activation='tanh'):
    # Neural network
    model = tf.keras.Sequential()
    # Input layer
    model.add(tf.keras.Input(3, name='Input')) # x,y,t
    # model.add(tf.keras.layers.InputLayer(3, name='Input'))
    # Hidden layers
    for i in range(num_hidden_layers):
        model.add(
            tf.keras.layers.Dense(num_neurons_per_layer,
            activation = tf.keras.activations.get(activation), 
            kernel_initializer = 'glorot_normal',
            name = 'Hidden_{}'.format(i+1)
            )
        )
    # Output layer
    model.add(tf.keras.layers.Dense(1, name='Output')) # T
    return model

def PDE(T, Tt, Tx, Ty, Txx, Tyy):
    return Tt -  - .5 * (Txx + Tyy)

def get_derivatives(model, X):
    x, y, t = tf.split(X, 3, axis=1)
    with tf.GradientTape(persistent = True) as t2:
        t2.watch(x)
        t2.watch(y)
        with tf.GradientTape(persistent = True) as t1:
            t1.watch(x)
            t1.watch(y)
            t1.watch(t)
            #T = model(X)
            T = model(tf.concat([x, y, t], axis=1))
        Tt = t1.gradient(T, t)
        Tx = t1.gradient(T, x)
        Ty = t1.gradient(T, y)
    Txx = t2.gradient(Tx, x)
    Tyy = t2.gradient(Ty, y)
    return T, Tt, Tx, Ty, Txx, Tyy

# Boudary conditions grad(T) . n = 0
def bc_1(model, X):
    _, _, _, Ty, _, _ = get_derivatives(model, X)
    return Ty - 0

def bc_2(model, X):
    _, _, Tx, _, _, _ = get_derivatives(model, X)
    return Tx - 0

def bc_3(model, X):
    _, _, _, Ty, _, _ = get_derivatives(model, X)
    return Ty - 0
    
def bc_4(model, X):
    _, _, Tx, _, _, _ = get_derivatives(model, X)
    return Tx - 0

def boundary_conditions(model, X_1, X_2, X_3, X_4):
    return bc_1(model, X_1), bc_2(model, X_2), bc_3(model, X_3), bc_4(model, X_4)

def custom_loss_m(model, X):
    m = PDE(*get_derivatives(model, X))
    loss_m = tf.reduce_mean(tf.square(m))
    return loss_m

def custom_loss_b(model, X_1, X_2, X_3, X_4):
    b1, b2, b3, b4 = boundary_conditions(model, X_1, X_2, X_3, X_4)
    loss_b = tf.reduce_mean(tf.square(b1) + tf.square(b2) + tf.square(b3) + tf.square(b4))
    return loss_b

def custom_loss_i(model, X):
    x, y, t = tf.split(X, 3, axis=1)
    T, T_0 = model(X), T0(x, y)
    loss_i = tf.reduce_mean(tf.square(T - T_0))
    return loss_i

def compute_loss(model, X, X_0, X_1, X_2, X_3, X_4):
    a, b, c = 1, 1, 1
    loss_m = custom_loss_m(model, X)
    loss_b = custom_loss_b(model, X_1, X_2, X_3, X_4)
    loss_i = custom_loss_i(model, X_0)
    loss = a * loss_m + b * loss_b + c * loss_i
    return loss, loss_m, loss_b, loss_i

def get_grad_loss(model, X, X_0, X_1, X_2, X_3, X_4):
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(model.trainable_variables)
        loss, loss_r, loss_m, loss_i = compute_loss(model, X, X_0, X_1, X_2, X_3, X_4)
    g = tape.gradient(loss, model.trainable_variables)
    return loss, g, loss_r, loss_m, loss_i

#Actualización de los pesos de model
@tf.function
def train_step(model, optimizer, X, X_0, X_1, X_2, X_3, X_4):
    loss, grad_theta, loss_r, loss_m, loss_i = get_grad_loss(model, X, X_0, X_1, X_2, X_3, X_4)
    #Realizar paso del descenso de gradiente
    optimizer.apply_gradients(zip(grad_theta, model.trainable_variables))
    return loss, loss_r, loss_m, loss_i

### Domain
# Nx, Ny, Nt = 128, 128, 1024
x_min, x_max = -5, 5
y_min, y_max = -5, 5
t_min, t_max = 0, 1
# Parameters
alpha = 1
# Data inside the domain
N_i = 1024
x_i = tf.random.uniform(shape=(N_i, 1), minval=x_min, maxval=x_max, dtype=DTYPE)
y_i = tf.random.uniform(shape=(N_i, 1), minval=y_min, maxval=y_max, dtype=DTYPE)
t_i = tf.random.uniform(shape=(N_i, 1), minval=t_min, maxval=t_max, dtype=DTYPE)
X_i = tf.concat([x_i, y_i, t_i], axis=1)
# Data on the boundary
N_b = 512
# First boundary
x_1 = tf.random.uniform(shape=(N_b, 1), minval=x_min, maxval=x_max, dtype=DTYPE)
y_1 = tf.constant(np.full((N_b, 1), y_min), dtype=DTYPE)
t_1 = tf.random.uniform(shape=(N_b, 1), minval=t_min, maxval=t_max, dtype=DTYPE)
X_1 = tf.concat([x_1, y_1, t_1], axis=1)
# Second boundary
# x_2 = tf.random.uniform(shape=(N_b, 1), minval=x_min, maxval=x_max, dtype=DTYPE)
x_2 = tf.constant(np.full((N_b, 1), x_max), dtype=DTYPE)
y_2 = tf.random.uniform(shape=(N_b, 1), minval=y_min, maxval=y_max, dtype=DTYPE)
t_2 = tf.random.uniform(shape=(N_b, 1), minval=t_min, maxval=t_max, dtype=DTYPE)
X_2 = tf.concat([x_2, y_2, t_2], axis=1)
# Third boundary
x_3 = tf.random.uniform(shape=(N_b, 1), minval=x_min, maxval=x_max, dtype=DTYPE)
# y_3 = tf.random.uniform(shape=(N_b, 1), minval=y_min, maxval=y_max, dtype=DTYPE)
y_3 = tf.constant(np.full((N_b, 1), y_max), dtype=DTYPE)
t_3 = tf.random.uniform(shape=(N_b, 1), minval=t_min, maxval=t_max, dtype=DTYPE)
X_3 = tf.concat([x_3, y_3, t_3], axis=1)
# Fourth boundary
# x_4 = tf.random.uniform(shape=(N_b, 1), minval=x_min, maxval=x_max, dtype=DTYPE)
x_4 = tf.constant(np.full((N_b, 1), x_min), dtype=DTYPE)
y_4 = tf.random.uniform(shape=(N_b, 1), minval=y_min, maxval=y_max, dtype=DTYPE)
t_4 = tf.random.uniform(shape=(N_b, 1), minval=t_min, maxval=t_max, dtype=DTYPE)
X_4 = tf.concat([x_4, y_4, t_4], axis=1)
# Initial condition
N_0 = 512
x_0 = tf.random.uniform(shape=(N_0, 1), minval=x_min, maxval=x_max, dtype=DTYPE)
y_0 = tf.random.uniform(shape=(N_0, 1), minval=y_min, maxval=y_max, dtype=DTYPE)
t_0 = tf.zeros(shape=(N_0, 1), dtype=DTYPE)
X_0 = tf.concat([x_0, y_0, t_0], axis=1)

# Run Model
if load_model_dir is None:
    model = init_model(num_hidden_layers=5, num_neurons_per_layer=100)
else:
    model = tf.keras.models.load_model(load_model_dir)
model.summary()

# Optimizer
# lr = 1e-2
# wd = None
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=wd)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=500,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Number of iterations
N_iter = 10000
# Early stopping
patience = 50
wait = 0
best = float('inf')
# Keep the best model
best_model = model
best_iter = 0
# Loss history
loss_history = np.zeros((N_iter, 4))
# Initial time
t0 = time.time()
# Training loop
for i in range(N_iter):
    loss, loss_m, loss_b, loss_i = train_step(model, optimizer, X_i, X_0, X_1, X_2, X_3, X_4)
    loss_history[i, :] = [loss.numpy(), loss_m.numpy(), loss_b.numpy(), loss_i.numpy()]
    # Loss log
    if i % 100 == 0:
        print(LOG_OUT.format(i, loss, loss_m, loss_b, loss_i))
    # Early stopping
    wait += 1
    if loss < best:
        best = loss
        best_model = model
        best_iter = i
        wait = 0
    if wait > patience:
        print('Early stopping at iteration {}'.format(i+1))
        break
print("Best loss: {:2.4e} at iteration: {}".format(best, best_iter+1))
loss_history = loss_history[:best_iter+1]
# Plot loss history
loss_plot(loss_history, log=True)
# Total time
print('\nComputation time: {} seconds\n'.format(round(time.time() - t0)))
print('Model name: {}'.format(model_name))
# Save model
best_model.save("{}/{}.h5".format(MODEL_DIR, model_name))
# Save loss history
np.savetxt('{}/{}.txt'.format(MODEL_DIR, model_name), loss_history)
