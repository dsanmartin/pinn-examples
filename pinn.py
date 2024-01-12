import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import function_factory
from plots import loss_plot

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only warnings and errors
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

# LOG_OUT = 'i: {:08d}: Total loss = {:2.4e}, Data loss = {:2.4e}, PDE loss = {:2.4e}, BC loss = {:2.4e}, IC loss = {:2.4e}'
LOG_OUT = 'i: {:08d}: Total loss = {:2.4e}, PDE loss = {:2.4e}, BC loss = {:2.4e}, IC loss = {:2.4e}'

class PINN:

    def __init__(self, model, pde, ic, bc, get_derivatives):
        self.model = model
        self.pde = pde
        self.ic = ic
        self.bc = bc
        self.get_derivatives = get_derivatives
        self.history = []

    def loss_data(self, X, y):
        U = self.model(X)
        loss = tf.reduce_mean(tf.square(U - y))
        return loss

    def loss_pde(self, X_m):
        m = self.pde(*self.get_derivatives(self.model, X_m))
        loss_m = 0
        for equation in m: # For each equation in the PDE
            loss_m = tf.reduce_mean(tf.square(equation))
        return loss_m

    def loss_bc(self, X_b):
        # b1, b2, b3, b4 = self.bc(self.model, X_b)
        # loss_b = 0
        # for bc1, bc2, bc3, bc4 in zip(b1, b2, b3, b4):
        #     loss_b += tf.reduce_mean(tf.square(bc1) + tf.square(bc2) + tf.square(bc3) + tf.square(bc4))
        # return loss_b
        loss_b = 0
        for i, bc in enumerate(self.bc):
            bci_loss = 0
            for bci in bc(self.model, X_b[i]):
                bci_loss += tf.square(bci)
            loss_b += bci_loss
        loss_b = tf.reduce_mean(loss_b)
        return loss_b

    def loss_ic(self, X_i):
        x, y, t = tf.split(X_i, 3, axis=1)
        U = self.model(X_i)
        # U0 = self.ic(x, y)
        loss_i = 0
        # for i, u0 in enumerate(U0): # For each initial condition
        #     loss_i += tf.reduce_mean(tf.square(U[:, i:i+1] - u0))
        for ic in self.ic:
            loss_i += tf.square(ic(x, y) - U)
        loss_i = tf.reduce_mean(loss_i)
        return loss_i

    def compute_loss(self, X, y):
        a, b, c, d = 1, 1, 1, 1 # Loss weights
        X_d, X_m, X_b, X_i = X
        loss_m = self.loss_pde(X_m)
        loss_b = self.loss_bc(X_b)
        loss_i = self.loss_ic(X_i)
        loss = b * loss_m + c * loss_b + d * loss_i
        # if X_d is not None:
        #     loss_d = self.loss_data(X_d, y)
        #     loss += a * loss_d
        # else:
        #     loss_d = None
        return loss, loss_m, loss_b, loss_i

    def get_grad_loss(self, X, y):
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(self.model.trainable_variables)
            loss, loss_m, loss_b, loss_i = self.compute_loss(X, y)
        g = tape.gradient(loss, self.model.trainable_variables)
        return g, loss, loss_m, loss_b, loss_i

    # Model's weights update
    @tf.function
    def train_step(self, optimizer, X, y):
        grad_theta, loss, loss_m, loss_b, loss_i = self.get_grad_loss(X, y)
        # Gradient descent step
        optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
        return loss, loss_m, loss_b, loss_i
    
    def adam_training(self, adam_params, X, y):
        # Adam optimizer
        N_iter = adam_params['N_iter'] #10000 # Number of iterations
        # Early stopping
        patience = adam_params['patience'] #100 # Number of iterations to wait before stopping
        wait = 0 # Counter
        best_loss = float('inf') # Best loss
        best_model = self.model # Best model
        best_iter = 0 # Best iteration
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=adam_params['lr'], #1e-2 # Learning rate
            decay_steps=adam_params['ds'], #500 # Decay steps
            decay_rate=adam_params['dr'] #0.9 # Decay rate
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # Training loop
        print("Training with Adam...")
        for i in range(N_iter):
            loss, loss_m, loss_b, loss_i = self.train_step(optimizer, X, y)
            # loss_history[i, :] = [loss.numpy(), loss_m.numpy(), loss_b.numpy(), loss_i.numpy()]
            # loss_d = loss_d.numpy() if loss_d is not None else -1
            self.history.append([loss.numpy(), loss_m.numpy(), loss_b.numpy(), loss_i.numpy()])
            # Loss log
            if i % 100 == 0:
                # print(self.history)
                print(LOG_OUT.format(i, *self.history[i]))
            # Early stopping
            wait += 1
            if loss < best_loss:
                best_loss = loss
                best_model = self.model
                best_iter = i
                wait = 0
            if wait > patience:
                print('Early stopping at iteration {}'.format(i+1))
                break
        print("Adam best loss: {:2.4e} at iteration: {}".format(best_loss, best_iter+1))
        self.model = best_model
        return None
    
    def lbfgs_training(self, lbfgs_params, X, y):
        # Based on piyueh's wrapper
        print("Training with L-BFGS...")
        func = function_factory(self.model, self.compute_loss, X, y)
        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, 
            initial_position=init_params, 
            parallel_iterations=lbfgs_params['pi'], # pi = 8 # Parallel iterations
            max_iterations=lbfgs_params['max_iter'] # max_iter = 500 # Maximum number of iterations
        )
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        loss_lbfgs = np.array(func.history)
        self.history = np.concatenate([np.array(self.history), loss_lbfgs], axis=0)
        print("L-BFGS best loss: {:2.4e}".format(loss_lbfgs[-1, 0]))
        # Plot loss history
        # loss_adam_lbfgs(loss_adam, loss_lbfgs, log=True)
        return None

    def training(self, adam_params, lbfgs_params, X, y):
        initial_time = time.time()
        # Training with adam
        if adam_params is not None:
            self.adam_training(adam_params, X, y)
        # Training with lbfgs
        if lbfgs_params is not None:
            self.lbfgs_training(lbfgs_params, X, y)
        final_time = time.time()
        print('\nComputation time: {} seconds\n'.format(round(final_time - initial_time)))
        # Plot loss history
        loss_plot(self.history, log=True)
        return None
    
    def save(self, path, domain):
        if path[-1] != '/':
            path += '/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save("{}model.h5".format(path))
        # Save loss history
        np.savetxt('{}loss.txt'.format(path), np.array(self.history))
        # Save domain in pickle file
        with open('{}domain.pkl'.format(path), 'wb') as f:
            pickle.dump(domain, f)
        return None