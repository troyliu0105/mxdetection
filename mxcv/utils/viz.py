from gluoncv.utils import viz


def save_net_plot(net, filename, shape=(1, 3, 416, 416)):
    viz.plot_network(net, shape=shape, save_prefix=filename)
