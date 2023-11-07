import numpy as np
import tensorflow as tf

def creat_domain_numpy(x_min, x_max, y_min, y_max, t_min, t_max, Nx, Ny, Nt):
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    t = np.linspace(t_min, t_max, Nt)
    X, Y = np.meshgrid(x, y)
    return X, Y, x, y, t

def create_axis(N_d, values, type='random'):
    if type == 'random':
        return tf.random.uniform(shape=(N_d, 1), minval=values[0], maxval=values[1], dtype=tf.float64)
    elif type == 'constant':
        return tf.constant(np.full((N_d, 1), values[0]), dtype=tf.float64)
    
def create_mesh(N_d, x_min, x_max, y_min, y_max):
    x = np.linspace(x_min, x_max, N_d)
    y = np.linspace(y_min, y_max, N_d)
    X, Y = np.meshgrid(x, y)
    X = tf.constant(X.flatten(), shape=(N_d ** 2, 1), dtype=tf.float64)
    Y = tf.constant(Y.flatten(), shape=(N_d ** 2, 1), dtype=tf.float64)
    return X, Y

def create_boundaries_data(N_b, x_min, x_max, y_min, y_max, t_min, t_max, type='random'):
    if type == 'random':
        # First boundary
        x_1 = create_axis(N_b, [x_min, x_max], type)
        y_1 = create_axis(N_b, [y_min], 'constant')    
        t_1 = create_axis(N_b, [t_min, t_max], type)
        # Second boundary
        x_2 = create_axis(N_b, [x_max], 'constant')
        y_2 = create_axis(N_b, [y_min, y_max], type)
        t_2 = create_axis(N_b, [t_min, t_max], type)
        # Third boundary
        x_3 = create_axis(N_b, [x_min, x_max], type)
        y_3 = create_axis(N_b, [y_max], 'constant')
        t_3 = create_axis(N_b, [t_min, t_max], type)
        # Fourth boundary
        x_4 = create_axis(N_b, [x_min], 'constant')
        y_4 = create_axis(N_b, [y_min, y_max], type)
        t_4 = create_axis(N_b, [t_min, t_max], type)
    else:
        # First boundary
        x_1, t_1 = create_mesh(N_b, x_min, x_max, t_min, t_max)
        y_1 = create_axis(N_b ** 2, [y_min], 'constant')
        # Second boundary
        y_2, t_2 = create_mesh(N_b, y_min, y_max, t_min, t_max)
        x_2 = create_axis(N_b ** 2, [x_max], 'constant')
        # Third boundary
        x_3, t_3 = create_mesh(N_b, x_min, x_max, t_min, t_max)
        y_3 = create_axis(N_b ** 2, [y_max], 'constant')
        # Fourth boundary
        y_4, t_4 = create_mesh(N_b, y_min, y_max, t_min, t_max)
        x_4 = create_axis(N_b ** 2, [x_min], 'constant')
    # Concatenate boundaries
    X_1 = tf.concat([x_1, y_1, t_1], axis=1)
    X_2 = tf.concat([x_2, y_2, t_2], axis=1)
    X_3 = tf.concat([x_3, y_3, t_3], axis=1)
    X_4 = tf.concat([x_4, y_4, t_4], axis=1)
    return X_1, X_2, X_3, X_4

def create_domain_data(N_i, x_min, x_max, y_min, y_max, t_min, t_max, type='random'):
    x = create_axis(N_i, [x_min, x_max], type)
    y = create_axis(N_i, [y_min, y_max], type)
    t = create_axis(N_i, [t_min, t_max], type)
    X = tf.concat([x, y, t], axis=1)
    return X

def create_initial_data(N_0, x_min, x_max, y_min, y_max, t_min, type='random'):
    if type == 'random':
        x = create_axis(N_0, [x_min, x_max], type)
        y = create_axis(N_0, [y_min, y_max], type)
        t = create_axis(N_0, [t_min], 'constant')
    else:
        x, y = create_mesh(N_0, x_min, x_max, y_min, y_max)
        t = create_axis(N_0 ** 2, [t_min], 'constant')
    X = tf.concat([x, y, t], axis=1)
    return X
    