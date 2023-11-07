import sys
import numpy as np
import tensorflow as tf
from plots import plot2D_2

# Model file path
model_file_path = sys.argv[1]

# Domain
Nx, Ny, Nt = 128, 128, 128
x_min, x_max = -5, 5
y_min, y_max = -5, 5
t_min, t_max = 0, 1
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)

# Load model
model = tf.keras.models.load_model(model_file_path)

# Crear una malla para x, y
X, Y = np.meshgrid(x, y)

# Create tensor to evaluate the model
X_eval_1 = tf.constant(np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_min)], axis=1), dtype=tf.float64)
X_eval_2 = tf.constant(np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_max)], axis=1), dtype=tf.float64)

# Evaluate model
model_predict_1 = model(X_eval_1).numpy()
model_predict_2 = model(X_eval_2).numpy()
T_predict_1 = model_predict_1.reshape(X.shape)
T_predict_2 = model_predict_2.reshape(X.shape)

# Plot
plot2D_2(x, y, T_predict_1, T_predict_2, r'$T(x, y, t={:2.4f})$'.format(t_min), r'$T(x, y, t={:2.4f})$'.format(t_max))