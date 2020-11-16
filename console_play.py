from dmdp.env.env import DMDPEnv
import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


LABEL_LIST = ["C", "P", "D"]


def render(G, pos):
    nx.draw(G, pos, with_labels=True)
    plt.show()


def add_node(G, current, next):
    G.add_edge(current, next)


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
    env = DMDPEnv(batch_size=1, n_clients=3, n_parkings=6)
    graph, times, status, mask = env.reset()
    graph = graph[0].numpy()
    labels = np.argmax(graph[:, 4:7], axis=1)
    labels = [LABEL_LIST[i] for i in labels]
    G, pos = reset_graph(graph, labels)
    is_terminal = tf.constant([False])
    while not is_terminal[0]:
        current = status[0][0].numpy()
        commands = list(map(int, input().split()))
        action = int(commands[0])
        if len(commands) > 1:
            verbose = commands[1]
        else:
            verbose = False
        action = tf.constant([action], dtype=tf.int32)
        [graph, time, status, mask], reward, is_terminal = env.step(action)
        add_node(G, labels[current] + str(current),
                 labels[commands[0]] + str(commands[0]))
        if verbose:
            render(G, pos)
