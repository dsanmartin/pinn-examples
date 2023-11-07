import sys
import pickle
import numpy as np
import tensorflow as tf
from plots import plot2D_2, streamplot_2

# Model file path
model_id = sys.argv[1]
path = 'models/' + model_id 
model_file_path = path + '_model.h5'
domain_file_path = path + '_domain.pkl'

# Domain
# Load domain from pickle file
with open(domain_file_path, 'rb') as f:
    domain = pickle.load(f)
x_min, x_max = domain['x_min'], domain['x_max']
y_min, y_max = domain['y_min'], domain['y_max']
t_min, t_max = domain['t_min'], domain['t_max']
Nx, Ny, Nt = 128, 128, 128
# x_min, x_max = -5, 5
# y_min, y_max = -5, 5
# t_min, t_max = 0, 1
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)

# Load model
model = tf.keras.models.load_model(model_file_path)
model.summary()

# Crear una malla para x, y
X, Y = np.meshgrid(x, y)

# Create tensor to evaluate the model
X_eval_1 = tf.constant(np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_min)], axis=1), dtype=tf.float64)
X_eval_2 = tf.constant(np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_max)], axis=1), dtype=tf.float64)

# Evaluate model
model_predict_1 = model(X_eval_1)#.numpy()
model_predict_2 = model(X_eval_2)#numpy()
u0, v0, p0 = model_predict_1[:, 0], model_predict_1[:, 1], model_predict_1[:, 2]
u1, v1, p1 = model_predict_2[:, 0], model_predict_2[:, 1], model_predict_2[:, 2]
u0 = u0.numpy().reshape(X.shape)
v0 = v0.numpy().reshape(X.shape)
p0 = p0.numpy().reshape(X.shape)
u1 = u1.numpy().reshape(X.shape)
v1 = v1.numpy().reshape(X.shape)
p1 = p1.numpy().reshape(X.shape)

# Plot
# plot2D_2(x, y, u0, u1, r'$u(x, y, t={:2.4f})$'.format(t_min), r'$u(x, y, t={:2.4f})$'.format(t_max))
# plot2D_2(x, y, v0, v1, r'$v(x, y, t={:2.4f})$'.format(t_min), r'$v(x, y, t={:2.4f})$'.format(t_max))
# plot2D_2(x, y, p0, p1, r'$p(x, y, t={:2.4f})$'.format(t_min), r'$p(x, y, t={:2.4f})$'.format(t_max))
streamplot_2(x, y, u0, v0, u1, v1, r'$u(x, y, t={:2.4f})$'.format(t_min), r'$u(x, y, t={:2.4f})$'.format(t_max))
