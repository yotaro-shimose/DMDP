"""
Delivery Markov Decision Process Simulator.
Once instances are created, all of the process can be compiled by tf.function using integer programming.
"""
import tensorflow as tf
from dmdp.functions import int_not, int_and, int_xor


class DMDP(tf.Module):
    """DMDP simulator class.
    Make sure simulator inherit tf.Module so that variables are properly managed.

    """

    def __init__(
            self,
            batch_size: int = 512,
            n_clients: int = 20,
            n_parkings: int = 40,
            walking_verocity: float = 0.1,
            vehicle_verocity: float = 1.,
            wait_penalty: float = 0.,
            delay_penalty: float = 100.,
            seed: int = None
    ):
        # random generator
        if seed is None:
            self.rand_generator = tf.random.Generator.from_non_deterministic_state()
        else:
            self.rand_generator = tf.random.Generator.from_seed(seed)
        # 1 for depo
        n_nodes = n_clients + n_parkings + 1
        # Per instance variables
        # GRAPH
        # B, N, 2 (2 for x, y)
        self.coordinates = tf.Variable(
            tf.zeros((batch_size, n_nodes, 2), dtype=tf.float32))
        # B, N, 2 (2 for time start and time end)
        self.time_constraints = tf.Variable(
            tf.zeros((batch_size, n_nodes, 2), dtype=tf.float32))
        # B, N
        self.parking_flags = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))
        # B, N
        self.depo_flags = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))
        # B, N
        self.client_flags = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))

        # Per step variables
        # B, N
        self.counts = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))
        # STATUS
        # B
        self.currents = tf.Variable(tf.zeros((batch_size, ), dtype=tf.int32))
        # B
        self.times = tf.Variable(tf.zeros((batch_size,), dtype=tf.float32))
        # B
        self.on_vehicles = tf.Variable(tf.ones((batch_size,), dtype=tf.int32))
        # B
        self.dones = tf.Variable(tf.zeros((batch_size,), dtype=tf.int32))
        # B
        self.vehicle_parked = tf.Variables(
            tf.zeros((batch_size), dtype=tf.int32))

        # scalars
        self.walking_verocity = tf.constant(walking_verocity, dtype=tf.float32)
        self.vehicle_verocity = tf.constant(vehicle_verocity, dtype=tf.float32)
        self.wait_penalty = tf.constant(wait_penalty, dtype=tf.float32)
        self.delay_penalty = tf.constant(delay_penalty, dtype=tf.float32)
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.n_clients = n_clients
        self.n_parkings = n_parkings

    def step(self, actions: tf.Tensor):
        """step function

        Args:
            actions (tf.Tensor): B
        """
        # B, N
        one_hot_actions = tf.one_hot(actions, depth=self.n_nodes)

        # B
        distances = tf.norm(self._get_cord(actions) -
                            self._get_cord(self.currents), axis=1)
        # B
        verocities = self.on_vehicles * self.vehicle_verocity + \
            int_not(self.on_vehicles) * self.walking_verocity
        # B
        times_elapsed = distances / verocities
        # B
        new_times = self.times + times_elapsed

        # Calculate wait and delay
        # B
        start = tf.reduce_sum(
            one_hot_actions * self.time_constrained[:, :, 0], axis=-1)
        # B
        waits = tf.maximum(tf.zeros(self.times.shape),
                           start - new_times)
        # B
        end = tf.reduce_sum(
            one_hot_actions * self.time_constrained[:, :, 1], axis=-1)
        # B
        delays = tf.maximum(tf.zeros(self.times.shape),
                            new_times - end)

        # Calculate cost
        # B
        cost = self.wait_penalty * waits + self.delay_penalty * delays + times_elapsed
        # Update counts
        self.counts.assign_add(one_hot_actions)
        # Update times
        self.times.assign(new_times)
        # Update current nodes
        self.currents.assign(actions)
        # Update vehicle parking nodes(if new node if on vehicle and previous node otherwise)
        self.vehicle_parked.assign(
            self.on_vehicles * actions + int_not(self.on_vehicles * self.vehicle_parked))

        # Update vehicle status (must be done after cost calculation!)
        # B
        status_change = tf.reduce_sum(
            self.parking_flags * one_hot_actions, axis=-1)
        self.on_vehicles.assign(int_xor(self.on_vehicles + status_change))

        # Calculate reward (0 for already done instances)
        # B
        rewards = int_not(self.done) * cost * (-1)

        # Calculate is_terminals
        # B, N
        non_parkings = self.depo_flags + self.client_flags
        # B
        is_terminals = tf.cast(tf.reduce_sum(
            self.counts * non_parkings, axis=1) - tf.reduce_sum(non_parkings, axis=1), tf.bool)
        # Update done (must be done after reward calculation)
        self.dones.assign(tf.cast(is_terminals, tf.int32))

        masks = self._get_mask()

        # Return graph, status, (don't forget stop_gradient!)
        # B, N, 3
        categories = tf.stack(
            [self.client_flags, self.parking_flags, self.depo_flags], axis=-1)
        # B, N, (2 + 2 + 3)
        graphs = tf.concat(
            [self.coordinates, self.time_constraints, categories], axis=-1)
        # B, 3
        status = tf.stack(
            [self.currents, self.times, self.on_vehicles], axis=-1)

        return [graphs, status, masks, rewards, is_terminals]

    def reset(self):
        self.coordinates.assign(tf.random.uniform(self.coordinates.shape))
        ones = tf.ones(self.client_flags.shape, dtype=tf.int32)
        zeros = tf.zeros(self.client_flags.shape, dtype=tf.int32)
        self.client_flags.assign(
            tf.concat([ones[:, :self.n_clients], zeros[:, self.n_clients:]], axis=1))
        self.parking_flags.assign(tf.concat([
            zeros[:, :self.n_clients],
            ones[:, self.n_clients:self.n_clients + self.n_parkings],
            zeros[:, self.n_clients + self.n_parkings:]
        ], axis=1))
        self.depo_flags.assign(tf.concat([
            zeros[:, :self.n_clients + self.n_parkings],
            ones[:, self.n_clients + self.n_parkings:]
        ], axis=1))
        self.time_constraints.assign(
            self._init_time_constraints(self.client_flags))
        # B, N
        self.counts.assign(
            tf.zeros((self.batch_size, self.n_nodes), dtype=tf.int32))
        # STATUS
        # B
        self.currents.assign(
            tf.ones((self.batch_size,), dtype=tf.int32) * (self.n_nodes - 1))
        # B
        self.times.assign(tf.zeros((self.batch_size,), dtype=tf.float32))
        # B
        self.on_vehicles.assign(tf.ones((self.batch_size,), dtype=tf.int32))
        # B
        self.dones.assign(tf.zeros((self.batch_size,), dtype=tf.int32))
        # B
        self.vehicle_parked.assign(
            tf.ones(self.batch_size, dtype=tf.int32) * (self.n_nodes - 1))

        # TODO Return States
        return

    def _get_mask(self):
        """Mask consisting of True for nodes you can visit next and False otherwise.
        Constraints are followings.
        1. You can not go to client nodes by vehicle.
        2. You can not go to parking node which you haven't parked vehicle by walk.
        3. You can not go to depo by walk.
        4. You can not go to depo when haven't visited all of the client nodes.
        5. You can not go to same parking node more than twice.
        6. You can not go to same client node more then once.

        all of sub masks are tensors with shape B and dtype tf.int32.

        """

        # 1.
        client_by_walk = int_not(int_and(self.client_flags, self.on_vehicle))
        # 2.
        never_leave_vehicle = int_not(
            int_and(int_not(self.vehicle_parked), self.parking_flags))
        # 3.
        bring_back_vehicle = int_not(
            int_and(self.depo_flags, int_not(self.on_vehicles)))
        # 4.
        finished_clients = int_not(tf.reduce_sum(
            self.counts * self.client_flags, axis=-1) - tf.reduce_sum(self.client_flags, axis=-1))
        finish_all_clients = int_not(self.depo_flags, finished_clients)
        # 5.

    def _get_cord(self, indices: tf.Tensor):
        """get cordinates corresponding indices

        Args:
            indices (tf.Tensor): B
        """
        indices = tf.one_hot(indices, depth=self.n_nodes)
        indices = tf.stack([indices, indices], axis=2)
        cordinates = tf.reduce_sum(indices * self.coordinates, axis=1)
        return cordinates

    def _init_time_constraints(self, client_flags):
        time_options = tf.constant([[1., 4.], [5., 7.], [7., 9.]])
        n_options = time_options.shape[0]
        time_options = tf.vectorized_map(self._broadcast, time_options)
        randint = tf.one_hot(self._randint(
            (self.batch_size, self.n_nodes), 0, n_options), depth=n_options, axis=0)
        randint = tf.expand_dims(randint, -1)
        randint = tf.tile(randint, [1, 1, 1, 2])
        return tf.reduce_sum(randint * time_options, axis=0) * \
            tf.expand_dims(tf.cast(client_flags, tf.float32), -1)

    @ tf.function
    def _broadcast(self, tensor):
        return tf.broadcast_to(tensor, shape=self.time_constraints.shape)

    def _randint(self, shape: tf.TensorShape, min: int, max: int):
        uniform = self.rand_generator.uniform(
            shape=shape, minval=min, maxval=max)
        return tf.cast(tf.floor(uniform), tf.int32)