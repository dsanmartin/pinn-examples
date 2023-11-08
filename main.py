import sys
import time
import tensorflow as tf
from pinn import PINN
from domain import create_boundaries_data, create_domain_data, create_initial_data
from example import model, PDE, get_derivatives, initial_conditions, boundary_conditions, domain, adam_params, lbfgs_params
from plots import plot3D, plot2D

def main():
    # Model name using date and time
    model_name = time.strftime('%Y%m%d%H%M%S')
    model_path = sys.argv[1] if len(sys.argv) > 1 else None # Load model from directory
    # Create data
    # Number of data points (per axis)
    N_i, N_b, N_0 = 16, 32, 32 # Inside the domain, on the boundary and initial condition
    # Data inside the domain
    x_min, x_max = domain['x_min'], domain['x_max']
    y_min, y_max = domain['y_min'], domain['y_max']
    t_min, t_max = domain['t_min'], domain['t_max']
    eps = 1e-3
    X_m = create_domain_data(N_i, x_min+eps, x_max-eps, y_min+eps, y_max-eps, t_min+eps, t_max, 'linspace') # N_i ** 3 points
    # Data on the boundary
    X_b = create_boundaries_data(N_b, x_min, x_max, y_min, y_max, t_min, t_max, 'linspace') # N_b ** 2 points per boundary
    # Initial condition
    X_0 = create_initial_data(N_0, x_min, x_max, y_min, y_max, t_min, 'linspace') # N_0 ** 2 points
    # Data 
    X_d = None
    y = None
    X = [X_d, X_m, X_b, X_0]
    # Define the model
    if model_path is not None:
        if model_path[-1] != '/':
            model_path += '/'
        nn = tf.keras.models.load_model(model_path + 'model.h5')
    else:
        model_path = 'models/' + model_name + '/'
        nn = model
    # Show model summary
    nn.summary()
    # Create PINN
    Pinn = PINN(nn, PDE, initial_conditions, boundary_conditions, get_derivatives)
    # Train the model
    Pinn.training(adam_params, lbfgs_params, X, y)
    # Save
    Pinn.save(model_path, domain)
    # Print model name
    print('Model name: {}'.format(model_name))
    

if __name__ == '__main__':
    main()