import matplotlib.pyplot as plt


def plotResult(title, xlabel, ylabel, xdata, ydata):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata, 'ro', markersize=2)
    plt.show()