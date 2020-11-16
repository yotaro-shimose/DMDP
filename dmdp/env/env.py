"""
Delivery Markov Decision Process Simulator.
Once instances are created, all of the process(step, reset) must be compiled by tf.function using
integer programming.
"""
import tensorflow as tf
from dmdp.modules.functions import int_not, int_and, int_xor


class DMDPEnv(tf.Module):
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
        # B, N
        self.last_masks = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))

        # Per step variables
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
        self.vehicle_parked = tf.Variable(
            tf.zeros((batch_size), dtype=tf.int32))

        self.state_dict = {
            'coordinates': self.coordinates,
            'time_constraints': self.time_constraints,
            'parking_flags': self.parking_flags,
            'depo_flags': self.depo_flags,
            'client_flags': self.client_flags,
            'last_masks': self.last_masks,
            'counts': self.counts,
            'currents': self.currents,
            'times': self.times,
            'on_vehicles': self.on_vehicles,
            'dones': self.dones,
            'vehicle_parked': self.vehicle_parked
        }

        # scalars
        self.walking_verocity = tf.constant(walking_verocity, dtype=tf.float32)
        self.vehicle_verocity = tf.constant(vehicle_verocity, dtype=tf.float32)
        self.wait_penalty = tf.constant(wait_penalty, dtype=tf.float32)
        self.delay_penalty = tf.constant(delay_penalty, dtype=tf.float32)
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.n_clients = n_clients
        self.n_parkings = n_parkings

    @tf.function
    def step(self, actions: tf.Tensor):
        """step function

        Args:
            actions (tf.Tensor): B
        """
        # B, N
        one_hot_actions = tf.one_hot(
            actions, depth=self.n_nodes, dtype=tf.int32)
        # B, N
        ones = tf.ones(one_hot_actions.shape, dtype=tf.int32)

        # filter actions by mask, which should not do anything
        # if actions are valid (unless it's done).
        # B, N
        filtered_actions = self._ignore_done(
            ones, self.last_masks * one_hot_actions)
        non_filtered_actions = self._ignore_done(ones, one_hot_actions)

        # Validate actions
        tf.assert_equal(filtered_actions, non_filtered_actions)

        # B
        distances = tf.norm(self._get_cord(actions) -
                            self._get_cord(self.currents), axis=1)
        # B
        verocities = tf.cast(self.on_vehicles, tf.float32) * self.vehicle_verocity + \
            tf.cast(int_not(self.on_vehicles), tf.float32) * \
            self.walking_verocity
        # B
        times_elapsed = distances / verocities
        # B
        new_times = self.times + times_elapsed

        # Calculate wait and delay
        # B
        start = tf.reduce_sum(
            tf.cast(one_hot_actions, tf.float32) * self.time_constraints[:, :, 0], axis=-1)
        # B
        waits = tf.maximum(tf.zeros(self.times.shape),
                           start - new_times)
        # B
        end = tf.reduce_sum(
            tf.cast(one_hot_actions, tf.float32) * self.time_constraints[:, :, 1], axis=-1)
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
            self.on_vehicles * actions + int_not(self.on_vehicles) * self.vehicle_parked)

        # Update vehicle status (must be done after cost calculation!)
        # B
        status_change = tf.reduce_sum(
            self.parking_flags * one_hot_actions, axis=-1)
        self.on_vehicles.assign(int_xor(self.on_vehicles, status_change))

        # Calculate reward (0 for already done instances)
        # B
        rewards = tf.cast(int_not(self.dones), tf.float32) * cost * (-1)

        # Calculate is_terminals
        # B
        finished_clients = tf.cast(tf.reduce_sum(self.counts * self.client_flags, -1) -
                                   tf.reduce_sum(self.client_flags, axis=-1) == 0, dtype=tf.int32)
        # B
        arrived_depo = tf.reduce_prod(
            tf.cast(self.depo_flags == one_hot_actions, tf.int32), axis=1)
        is_terminals = int_and(finished_clients, arrived_depo)
        # Update done (must be done after reward calculation)
        self.dones.assign(tf.cast(is_terminals, tf.int32))

        # (B, N), (B,), (B, N, 2 + 2 + 3 + 1), (B, 3)
        graphs, times, status, masks = self.get_states()
        self.last_masks.assign(masks)

        return [[graphs, times, status, masks], rewards, is_terminals]

    @tf.function
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

        # (B, N), (B,), (B, N, 2 + 2 + 3 + 1), (B, 3)
        states = _, _, _, masks = self.get_states()
        self.last_masks.assign(masks)
        return states

    def get_states(self):
        # B, N
        masks = self._get_mask()

        # B, N, (2 + 2 + 3 + 1)
        graphs = self._get_graph()

        # B, 3
        status = self._get_status()

        return [graphs, tf.identity(self.times), status, masks]

    def _get_graph(self):
        # B, N, 3
        categories = tf.stack(
            [self.client_flags, self.parking_flags, self.depo_flags], axis=-1)
        categories = tf.cast(categories, tf.float32)
        # B, N
        count_one = self.counts >= 1
        count_two = self.counts >= 2
        # B, N, 2
        counts = tf.stack([count_one, count_two], axis=-1)
        counts = tf.cast(counts, tf.float32)
        # B, N, (2 + 2 + 3 + 1)
        graphs = tf.concat(
            [self.coordinates, self.time_constraints, categories, counts], axis=-1)
        return graphs

    def _get_status(self):
        # B, 3
        status = tf.stack(
            [self.currents, self.on_vehicles, self.vehicle_parked], axis=-1)
        return status

    def _get_mask(self):
        """Mask consisting of True for nodes you can visit next and False otherwise.
        Constraints are followings.
        1. You can not go to client nodes by vehicle.
        2. You can not go to parking node which you haven't parked vehicle by walk.
        3. You can not go to depo by walk.
        4. You can not go to depo when haven't visited all of the client nodes.
        5. You can not go to same parking node more than twice.
        6. You can not go to same client node more then once.
        7. You can not stay on a same node twice in a row.

        all of sub masks are tensors with shape B and dtype tf.int32.

        """

        # 1.
        client_by_walk = int_not(
            int_and(self.client_flags, tf.expand_dims(self.on_vehicles, -1)))
        # 2.
        # parking node without vehicle
        # B, N
        non_vehicle_parkings = int_and(self.parking_flags, int_not(
            tf.one_hot(self.vehicle_parked, depth=self.n_nodes, dtype=tf.int32)))
        never_leave_vehicle = int_not(int_and(tf.expand_dims(int_not(self.on_vehicles), -1),
                                              int_and(self.parking_flags, non_vehicle_parkings)))
        # 3.
        bring_back_vehicle = int_not(
            int_and(self.depo_flags, tf.expand_dims(int_not(self.on_vehicles), -1)))
        # 4.
        not_finished_clients = tf.cast(tf.reduce_sum(
            self.counts * self.client_flags, -1) - tf.reduce_sum(self.client_flags, axis=-1) < 0,
            dtype=tf.int32)
        finish_all_clients = int_not(
            int_and(self.depo_flags, tf.expand_dims(not_finished_clients, -1)))
        # 5.
        never_park_twice = int_not(int_and(
            self.parking_flags, tf.cast(self.counts >= 2, tf.int32)))
        # 6.
        never_deliver_twice = int_not(
            int_and(self.client_flags, tf.cast(self.counts >= 1, tf.int32)))
        # 7.
        never_stay = int_not(tf.one_hot(
            self.currents, depth=self.n_nodes, dtype=tf.int32))

        return\
            client_by_walk *\
            never_leave_vehicle *\
            bring_back_vehicle *\
            finish_all_clients *\
            never_park_twice *\
            never_deliver_twice *\
            never_stay

    def _get_cord(self, indices: tf.Tensor):
        """get cordinates corresponding to indices

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
        return tf.reduce_sum(randint * time_options, axis=0) *\
            tf.expand_dims(tf.cast(client_flags, tf.float32), -1)

    @ tf.function
    def _broadcast(self, tensor):
        return tf.broadcast_to(tensor, shape=self.time_constraints.shape)

    def _randint(self, shape: tf.TensorShape, min: int, max: int):
        uniform = self.rand_generator.uniform(
            shape=shape, minval=min, maxval=max)
        return tf.cast(tf.floor(uniform), tf.int32)

    def _ignore_done(self, target: tf.Tensor, assignment: tf.Tensor):
        """_ignore_done returns tensor filled with target's value for done instances
        and with assignment's value for undone instances

        Args:
            target (tf.Tensor): value to fill done instances
            assignment (tf.Tensor): value to fill undone instances

        Returns:
            [type]: [description]
        """

        if tf.rank(self.dones) < tf.rank(target):
            dones = tf.expand_dims(self.dones, -1)
        else:
            dones = self.dones

        return target * dones + assignment * int_not(dones)

    def import_states(
        self,
        states: dict
    ):
        # Per instance variables
        # B, N, 2 (2 for x, y)
        self.coordinates.assign(states['coordinates'])
        # B, N, 2 (2 for time start and time end)
        self.time_constraints.assign(states['time_constraints'])
        # B, N
        self.parking_flags.assign(states['parking_flags'])
        # B, N
        self.depo_flags.assign(states['depo_flags'])
        # B, N
        self.client_flags.assign(states['client_flags'])
        # B, N
        self.last_masks.assign(states['last_masks'])

        # Per step variables
        self.counts.assign(states['counts'])
        # STATUS
        # B
        self.currents.assign(states['currents'])
        # B
        self.times.assign(states['times'])
        # B
        self.on_vehicles.assign(states['on_vehicles'])
        # B
        self.dones.assign(states['dones'])
        # B
        self.vehicle_parked.assign(states['vehicle_parked'])

    def export_states(self):
        return {
            'coordinates': tf.identity(self.coordinates),
            'time_constraints': tf.identity(self.time_constraints),
            'parking_flags': tf.identity(self.parking_flags),
            'depo_flags': tf.identity(self.depo_flags),
            'client_flags': tf.identity(self.client_flags),
            'last_masks': tf.identity(self.last_masks),
            'counts': tf.identity(self.counts),
            'currents': tf.identity(self.currents),
            'times': tf.identity(self.times),
            'on_vehicles': tf.identity(self.on_vehicles),
            'dones': tf.identity(self.dones),
            'vehicle_parked': tf.identity(self.vehicle_parked)
        }
