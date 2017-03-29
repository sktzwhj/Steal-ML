import matplotlib.pyplot as plt
import matplotlib
import numpy
import cPickle


fd_cost = open("../neural-nets/cost_list", 'r')

fd_mean = open("../neural-nets/mean_list", 'r')

cost_list = cPickle.loads(fd_cost.read())

mean_list = cPickle.loads(fd_mean.read())

plt.plot(cost_list, color = 'r')

plt.plot(mean_list, color = 'b')

plt.show()