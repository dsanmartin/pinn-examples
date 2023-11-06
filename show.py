import sys
import numpy as np
import tensorflow as tf
from plots import plot2D#, loss_plot

n_args = len(sys.argv)
print(n_args)
model_file_path = sys.argv[1]
loss_file_path = None
if n_args > 2:
    loss_file_path = sys.argv[2]

Nx, Ny, Nt = 32, 32, 32
x_min, x_max = -5, 5
y_min, y_max = -5, 5
t_min, t_max = 0, 1
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)

# Load loss
# if loss_file_path is not None:
#     loss = np.loadtxt(loss_file_path)
#     loss_plot(loss, log=True)

# Load model
model = tf.keras.models.load_model(model_file_path)

# n = 0 # Time step to evaluate
# te = t[n] # time evaluation

# Crear una malla para x, y
X, Y = np.meshgrid(x, y)

# Create tensor to evaluate the model
X_eval = np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_min)], axis=1) 
X_eval = tf.constant(X_eval, dtype=tf.float64)
# Evaluar el modelo
model_predict = model(X_eval).numpy()
T_predict = model_predict.reshape(X.shape)
plot2D(x, y, T_predict, r'$T(x, y, t=0)$')

X_eval = np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_max)], axis=1) 
X_eval = tf.constant(X_eval, dtype=tf.float64)
# Evaluar el modelo
model_predict = model(X_eval).numpy()
T_predict = model_predict.reshape(X.shape)
plot2D(x, y, T_predict, r'$T(x, y, t={})$'.format(t_max))
