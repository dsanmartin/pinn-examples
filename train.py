import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from plots import loss_adam_lbfgs
from utils import function_factory
from domain import create_boundaries_data, create_domain_data, create_initial_data

# NUM_THREADS = 4
LOG_OUT = 'i: {:08d}: Total loss = {:2.4e}, PDE loss = {:2.4e}, BC loss = {:2.4e}, IC loss = {:2.4e}'
MODEL_DIR = 'models'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
# If you want to limit the number of threads
# os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
# os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
# os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)
# tf.config.threading.set_inter_op_parallelism_threads(
#     NUM_THREADS
# )
# tf.config.threading.set_intra_op_parallelism_threads(
#     NUM_THREADS
# )
# tf.config.set_soft_device_placement(True)
tf.keras.backend.set_floatx('float64')

# Model name using date and time
model_name = time.strftime('%Y%m%d%H%M%S')
load_model_dir = sys.argv[1] if len(sys.argv) > 1 else None # Load model from directory

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

# PDE T_t = alpha * (T_xx + T_yy)
def PDE(T, Tt, Tx, Ty, Txx, Tyy):
    return Tt - .5 * (Txx + Tyy)

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

# Model's weights update
@tf.function
def train_step(model, optimizer, X, X_0, X_1, X_2, X_3, X_4):
    loss, grad_theta, loss_r, loss_m, loss_i = get_grad_loss(model, X, X_0, X_1, X_2, X_3, X_4)
    #Realizar paso del descenso de gradiente
    optimizer.apply_gradients(zip(grad_theta, model.trainable_variables))
    return loss, loss_r, loss_m, loss_i

### Domain ###
# Nx, Ny, Nt = 128, 128, 1024
x_min, x_max = -5, 5
y_min, y_max = -5, 5
t_min, t_max = 0, 1
# Number of data points
N_i, N_b, N_0 = 1024, 512, 512 # Inside the domain, on the boundary and initial condition
# Data inside the domain
X_i = create_domain_data(N_i, x_min, x_max, y_min, y_max, t_min, t_max)
# Data on the boundary
X_1, X_2, X_3, X_4 = create_boundaries_data(N_b, x_min, x_max, y_min, y_max, t_min, t_max)
# Initial condition
X_0 = create_initial_data(N_0, x_min, x_max, y_min, y_max, t_min)

# Create/load model
if load_model_dir is None: # New model
    model = init_model() #num_hidden_layers=5, num_neurons_per_layer=100)
else: # Load model
    model = tf.keras.models.load_model(load_model_dir)
model.summary() # Show configuration

# Optimization. First using Adam, then L-BFGS #
# Adam optimizer
N_iter = 10000 # Number of iterations
# Early stopping
patience = 50 # Number of iterations to wait before stopping
wait = 0 # Counter
best_loss = float('inf') # Best loss
best_model = model # Best model
best_iter = 0 # Best iteration
loss_history = np.zeros((N_iter, 4)) # Loss history
lr = 1e-2 # Learning rate
ds = 500 # Decay steps
dr = 0.9 # Decay rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=ds,
    decay_rate=dr
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# Initial time
t0 = time.time()
# Training loop
print("Training with Adam...")
for i in range(N_iter):
    loss, loss_m, loss_b, loss_i = train_step(model, optimizer, X_i, X_0, X_1, X_2, X_3, X_4)
    # loss, loss_m, loss_b, loss_i = train_step_lbfgs(model, X_i, X_0, X_1, X_2, X_3, X_4)
    loss_history[i, :] = [loss.numpy(), loss_m.numpy(), loss_b.numpy(), loss_i.numpy()]
    # Loss log
    if i % 100 == 0:
        print(LOG_OUT.format(i, loss, loss_m, loss_b, loss_i))
    # Early stopping
    wait += 1
    if loss < best_loss:
        best_loss = loss
        best_model = model
        best_iter = i
        wait = 0
    if wait > patience:
        print('Early stopping at iteration {}'.format(i+1))
        break
# Best loss for adam
print("Adam best loss: {:2.4e} at iteration: {}".format(best_loss, best_iter+1))
loss_adam = loss_history[:best_iter+1]
# Second part using L-BFGS. Based on piyueh's wrapper
pi = 8 # Parallel iterations
max_iter = 5000 # Max iterations
print("Training with L-BFGS...")
func = function_factory(best_model, compute_loss, X_i, X_0, X_1, X_2, X_3, X_4)
# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(func.idx, best_model.trainable_variables)
# train the model with L-BFGS solver
results = tfp.optimizer.lbfgs_minimize(
    value_and_gradients_function=func, 
    initial_position=init_params, 
    parallel_iterations=pi,
    max_iterations=max_iter
)
# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
func.assign_new_model_parameters(results.position)
loss_lbfgs = np.array(func.history)
print("L-BFGS best loss: {:2.4e}".format(loss_lbfgs[-1, 0]))
# Plot loss history
loss_adam_lbfgs(loss_adam, loss_lbfgs, log=True)
# Total time
print('\nComputation time: {} seconds\n'.format(round(time.time() - t0)))
print('Model name: {}'.format(model_name))
# Save model
best_model.save("{}/{}.h5".format(MODEL_DIR, model_name))
# Save loss history
np.savetxt('{}/{}_loss_adam.txt'.format(MODEL_DIR, model_name), loss_adam)
np.savetxt('{}/{}_loss_lbfgs.txt'.format(MODEL_DIR, model_name), loss_lbfgs)
