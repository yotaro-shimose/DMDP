from dmdp.modules.models.network import GraphAttentionNetwork
from dmdp.reinforce.reinforce import Reinforce


if __name__ == '__main__':
    d_model = 128
    d_key = 16
    n_heads = 8
    depth = 2
    th_range = 10
    weight_balancer = 0.12

    def network_builder():
        return GraphAttentionNetwork(d_model, d_key, n_heads, depth, th_range, weight_balancer)

    reinforce = Reinforce(network_builder)
    reinforce.start()
