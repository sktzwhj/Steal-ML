import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cPickle
import sklearn.metrics as metrics


fd_cluster_path = open("../neural-nets/cluster_path", 'r')
fd_costs = open("../neural-nets/cost_list", 'r')

cost_list = cPickle.loads(fd_cluster_path.read())
costs = cPickle.loads(fd_costs.read())
mutual_info = []
costs_data = []

for i in cost_list:
    mi = metrics.normalized_mutual_info_score(np.array(i), np.array(cost_list[-1]))
    mutual_info.append(mi)

for j in costs:
    costs_data.append(j)


plt.plot(mutual_info, marker='x', color='b', label="normalized mutual info score for weight clustering")
plt.plot(costs_data, marker='o', color='r', label = "loss")
plt.legend().draggable()
plt.show()