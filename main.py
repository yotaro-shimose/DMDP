from dmdp.env import DMDP
import tensorflow as tf


if __name__ == '__main__':
    env = DMDP(batch_size=4, n_clients=3, n_parkings=6)
    shape = (3, 3)
    min = 4,
    max = 6 + 1
    env.reset()
    env.step()
