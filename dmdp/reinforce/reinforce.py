from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp
from dmdp.env.env import DMDPEnv
from dmdp.modules.functions import int_not, sample_action


@tf.function
def ttest_smaller(x: tf.Tensor, y: tf.Tensor, significance=tf.constant(0.05)):
    """ttest_smaller returns True when mean of x is significantly smaller than mean of y.

    Args:
        x (tf.Tensor): Rank 1 tensor
        y (tf.Tensor): A tensor with the same shape of x.
        significance (tf.Tensor, optional, dtype=tf.float32): Probability of unintended return of
        True when mean of populations are the same.

    Returns:
        [type]: [description]
    """
    tf.assert_equal(x.shape, y.shape)
    tf.assert_rank(x, 1)
    square_sum = tf.reduce_sum(tf.square(x - tf.reduce_mean(x, keepdims=True))) + \
        tf.reduce_sum(tf.square(x - tf.reduce_mean(x, keepdims=True)))
    sigma = square_sum / tf.cast(2 * tf.size(x) - 2, tf.float32)
    t = (tf.reduce_mean(x) - tf.reduce_mean(y)) / \
        (tf.cast(tf.sqrt(2 / tf.size(x)), tf.float32) * sigma)
    df = tf.cast(2 * tf.size(x) - 2, tf.float32)
    student = tfp.distributions.StudentT(df=df, loc=0, scale=1)
    p_value = student.cdf(t)
    return p_value < significance


class Reinforce:
    def __init__(
        self,
        network_builder: Callable,
        n_epochs: int = 10000,
        n_iterations: int = 10,
        n_validations: int = 100,
        n_parallels: int = 5,
        n_nodes_min: int = 14,
        n_nodes_max: int = 14,
        learning_rate: float = 9e-5,
        significance: float = 0.05,
        logger=None,
        save_dir="./models/",
        load_dir=None,
    ):
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.n_validations = n_validations
        self.n_parallels = n_parallels
        self.online_env = DMDPEnv()
        self.baseline_env = DMDPEnv()
        self.n_nodes_min = n_nodes_min
        self.n_nodes_max = n_nodes_max
        self.save_dir = save_dir

        self.online_network: tf.keras.models.Model = network_builder()
        self.baseline_network: tf.keras.models.Model = network_builder()
        if load_dir:
            self.load(load_dir)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.significance = significance
        self.logger = logger

    def start(self):
        for epoch in range(self.n_epochs):
            for iteration in range(self.n_iterations):
                # start = time.time()
                metrics = self.train_on_episode()
                # end = time.time()
                # print(f"経過時間: {end - start}(s)")
                step = epoch * self.n_iterations + iteration
                self.logger.log(metrics, step)
            if self.validate():
                print("Validation passed")
                if self.save_dir:
                    self.save(self.save_dir)
                self.synchronize(self.online_network,
                                 self.baseline_network)

    # @tf.function
    def train_on_episode(self):
        """train_on_episode executes parallel episodes at the same time and learn from the experiences.
        """
        # ** Initialization ** #

        # Initialize state
        self.online_env.reset()
        # Copy env list for baseline.
        self.baseline_env.import_states(self.online_env.export_states())

        with tf.GradientTape() as tape:
            # Greedy rollout
            base_rewards, _ = self.play_game(
                env=self.baseline_env,
                network=self.baseline_network,
                greedy=True
            )

            # Execute an episode for each online environment
            online_rewards, log_likelihood = self.play_game(
                env=self.online_env,
                network=self.online_network,
                greedy=False
            )

            # ** Learn from experience ** #

            trainable_variables = self.online_network.trainable_variables
            excess_cost = tf.stop_gradient((base_rewards - online_rewards))
            # Get policy gradient to apply to our network
            policy_gradient = tape.gradient(tf.reduce_mean(
                excess_cost * log_likelihood, trainable_variables))

            # Apply gradient
            self.optimizer.apply_gradients(
                zip(policy_gradient, trainable_variables))
        # metrics
        metrics = {
            "cost against baseline": tf.reduce_mean(excess_cost),
            "base_rewards": tf.reduce_mean(base_rewards),
            "online_rewards": tf.reduce_mean(online_rewards)
        }
        return metrics

    # @tf.function
    def play_game(
        self,
        env: DMDPEnv,
        network: tf.keras.models.Model,
        greedy: tf.Tensor = tf.constant(False)
    ):
        """play games in parallels

        Args:
            envs (tf.Module): list of environments which are RESET.
            network (tf.keras.models.Model): [description]
            greedy (bool, optional): [description]. Defaults to False.

        Returns:
            tuple(tf.Tensor(batch_size), tf.Tensor(batch_size, graph_size, graph_size)):
                rewards, policies
        """

        # ** Initialization ** #
        # Get graph
        dones = tf.zeros((self.n_parallels,), dtype=tf.int32)
        ones = tf.ones(dones.shape, dtype=tf.int32)
        rewards = tf.zeros(dones.shape, dtype=tf.float32)
        log_likelihood = tf.zeros(dones.shape, dtype=tf.float32)
        states = env.get_states()
        divisor = tf.zeros(dones.shape, dtype=tf.float32)
        # Note AutoGraph can't change tensor shape and dtype in while loop
        while tf.math.logical_not(tf.reduce_all(dones == ones)):
            # Get policy
            # B, N
            policies = network(states)

            # Determine actions to take
            if greedy:
                actions = tf.argmax(policies, axis=1)
            else:
                actions = sample_action(policies, axis=1)
            # Filter to ignore probabilities of actions which won't be taken
            # B, N
            indices = tf.one_hot(
                actions, depth=policies.shape[-1], dtype=tf.float32)
            # Probabilities of choosing the actions
            # B
            sample_log_probability = tf.math.log(
                tf.reduce_sum(indices * policies, axis=-1))

            # Average over log probabilities of sampling the actions
            # B
            new_divisor = divisor + tf.cast(int_not(dones), tf.float32)
            # B
            update = (log_likelihood / new_divisor) * divisor + \
                sample_log_probability / new_divisor - log_likelihood
            # B
            divisor = tf.identity(new_divisor)

            # Calculate average of log probabilities for undone instances
            # B
            log_likelihood = log_likelihood + tf.where(
                dones == ones,
                tf.zeros(log_likelihood.shape, dtype=tf.float32),
                update
            )

            states, new_rewards, dones = env.step(actions)
            rewards = rewards + \
                tf.where(dones == ones, tf.zeros(
                    dones.shape, dtype=tf.float32), new_rewards)

        return rewards, log_likelihood

    def synchronize(self, original: tf.keras.models.Model, target: tf.keras.models.Model):
        target.set_weights(original.get_weights)

    def validate(self):

        # ** Initialization ** #

        # Initialize state
        # Initialize state
        self.online_env.reset()
        # Copy env list for baseline
        self.baseline_env.import_states(self.online_env.export_states())
        online_rewards, _ = self.play_game(
            self.online_env, self.online_network, greedy=True)
        base_rewards, _ = self.play_game(
            self.baseline_env, self.baseline_network, greedy=True)

        return ttest_smaller(online_rewards, base_rewards, self.significance)

    def save(self, path):
        self.online_network.save_weights(path)

    def load(self, path):
        self.online_network.load_weights(path)
        self.baseline_network.load_weights(path)

    def demo(self, graph_size=None):
        raise NotImplementedError
