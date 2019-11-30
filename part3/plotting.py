import matplotlib.pyplot as plt


def plotResult(title, xlabel, ylabel, xdata, ydata):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata, 'ro', markersize=2, label="State (0,0)")
    plt.legend()
    plt.show()


def multiplot(title, xlabel, ylabel, xdata, ydata, legends=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legends != None:
        plt.legend(legends)
        for i in range(len(ydata)):
            plt.plot(xdata, ydata[i], markersize=2, label=legends[i])
    else:
        for i in range(len(ydata)):
            plt.plot(xdata, ydata[i], markersize=2, label=legends[i])

    plt.legend()
    plt.show()