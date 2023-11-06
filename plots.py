import numpy as np
import matplotlib.pyplot as plt

def plot2D(x, y, z, title=None):
    # Graficar
    plt.figure(figsize=(7, 6))
    cp = plt.contourf(x, y, z, cmap='viridis')
    plt.colorbar(cp)
    if title is not None:
        plt.title(title)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()

def plot3D(x, y, z):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', alpha=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    plt.show()
    return None

def plot2D_2(x, y, z1, z2, tit1="", tit2=""):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cp1 = ax[0].contourf(x, y, z1, cmap='jet')
    cp2 = ax[1].contourf(x, y, z2, cmap='jet')
    ax[0].set_xlabel(r'$x$')
    ax[1].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_title(tit1)
    ax[1].set_title(tit2)
    fig.colorbar(cp1, ax=ax[0])
    fig.colorbar(cp2, ax=ax[1])
    plt.tight_layout()
    plt.show()


def loss_plot(loss_history, log=False):
    plt.figure(figsize=(7, 6))
    plt.plot(loss_history[:, 0], label=r'Loss')
    plt.plot(loss_history[:, 1], label=r'Loss model')
    plt.plot(loss_history[:, 2], label=r'Loss bc')
    plt.plot(loss_history[:, 3], label=r'Loss ic')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if log:
        plt.yscale('log')
    plt.show()
    return None

def loss_adam_lbfgs(loss_adam, loss_lbfgs, log=True):
    total_loss = np.concatenate((loss_adam, loss_lbfgs))
    plt.figure(figsize=(7, 6))
    plt.plot(total_loss[:, 0], label=r'Loss')
    plt.plot(total_loss[:, 1], label=r'Loss model')
    plt.plot(total_loss[:, 2], label=r'Loss bc')
    plt.plot(total_loss[:, 3], label=r'Loss ic')
    plt.axvline(x = len(loss_adam), color = 'k', linestyle='--', label = 'Optimizer change')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if log:
        plt.yscale('log')
    plt.show()
    return None
