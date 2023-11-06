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