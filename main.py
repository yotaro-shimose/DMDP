from dmdp.env import DMDP
import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


LABEL_LIST = ["C", "P", "D"]


def render(G, pos, current, next):
    G.add_edge(current, next)
    nx.draw(G, pos, with_labels=True)
    plt.show()


def reset_graph(graph, labels):
    G = nx.MultiDiGraph()
    #
    pos = graph[:, 0:2]
    posdic = {}
    for i in range(len(graph)):
        label = labels[i]
        label = label + str(i)
        G.add_node(label)
        posdic[label] = pos[i]
    return G, posdic


if __name__ == '__main__':
    env = DMDP(batch_size=1, n_clients=3, n_parkings=6)
    graph, times, status, mask = env.reset()
    graph = graph[0].numpy()
    labels = np.argmax(graph[:, 4:7], axis=1)
    labels = [LABEL_LIST[i] for i in labels]
    G, pos = reset_graph(graph, labels)
    is_terminal = tf.constant([False])
    while not is_terminal[0]:
        current = status[0][0].numpy()
        action = int(input("come on!"))
        render(G, pos, labels[current] + str(current),
               labels[action] + str(action))
        action = tf.constant([action], dtype=tf.int32)
        graph, time, status, mask, reward, is_terminal = env.step(action)
