"""DMDPEnv test module.
DO NOT forget uncommenting tf.function out on step and reset!

"""


from dmdp.env.env import DMDPEnv
import tensorflow as tf


def raise_error_at_last(env, actions):
    graph, times, status, mask = env.reset()
    answer = False
    for i in range(len(actions)):
        try:
            action = actions[i]
            action = tf.constant([action], dtype=tf.int32)
            graph, time, status, mask, reward, is_terminal = env.step(action)
        except tf.errors.InvalidArgumentError:
            if i == len(actions) - 1:
                answer = True
            else:
                break
    return answer


def complete_synario(env, actions):
    graph, time, status, mask = env.reset()
    answer = False
    for i in range(len(actions)):
        try:
            action = actions[i]
            action = tf.constant([action], dtype=tf.int32)
            graph, time, status, mask, reward, is_terminal = env.step(action)
        except tf.errors.InvalidArgumentError:
            break
    else:
        answer = True
    return answer


def test_env_mask():
    env = DMDPEnv(batch_size=1, n_clients=3, n_parkings=6)
    graph, times, status, mask = env.reset()
    # 1. You can not go to client nodes by vehicle.
    actions = [5, 1, 5, 0]
    assert raise_error_at_last(env, actions)
    # 2. You can not go to parking node which you haven't parked vehicle by walk.
    actions = [8, 1, 7]
    assert raise_error_at_last(env, actions)
    # 3. You can not go to depo by walk.
    actions = [5, 0, 1, 2, 9]
    assert raise_error_at_last(env, actions)
    # 4. You can not go to depo when haven't visited all of the client nodes.
    actions = [7, 0, 2, 7, 9]
    assert raise_error_at_last(env, actions)
    # 5. You can not go to same parking node more than twice.
    actions = [7, 0, 7, 3, 1, 3, 7]
    assert raise_error_at_last(env, actions)
    # 6. You can not go to same client node more then once.
    actions = [4, 0, 4, 3, 0]
    assert raise_error_at_last(env, actions)
    # 7. You can not stay on a same node twice in a row.
    actions = [3, 1, 1]
    assert raise_error_at_last(env, actions)


def test_env_complete():
    env = DMDPEnv(batch_size=1, n_clients=3, n_parkings=6)
    actions = [4, 0, 1, 2, 4, 9]
    assert complete_synario(env, actions)
    actions = [4, 0, 4, 3, 1, 3, 5, 2, 5, 9]
    assert complete_synario(env, actions)
    actions = [7, 2, 7, 8, 1, 0, 8, 9]
    assert complete_synario(env, actions)
