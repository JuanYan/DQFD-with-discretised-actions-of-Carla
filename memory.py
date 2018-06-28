# -*- coding: utf-8 -*-
"""
RL replay memory
"""
import numpy as np

# set seed, so the results are repeatable
np.random.seed(0)


class SumTree:
    def __init__(self, capacity, permanent_size=0):
        """
        A Sum Tree to store data with priority
        Leaf stores priority of data, and each parent node is the sum of child nodes
        :param capacity: total number of data to store
        :param permanent_size: data size to keep from beginning if override
        """
        self.full = False
        self.data_index = 0
        self.capacity = capacity
        self.priority_tree = np.zeros(2 * capacity - 1)  # stores priorities in leaf
        self.data = np.zeros(capacity, dtype=object)  # stores data of objects
        self.permanent_size = permanent_size  # numbers of data which never be replaced
        if not 0 <= self.permanent_size <= self.capacity:
            raise RuntimeError('permanent data shall be inside capacity!')

    def __len__(self):
        return self.capacity if self.full else self.data_index

    def append(self, priority, data):
        """
        add new data with priority to tree
        :param priority: priority
        :param data: data value
        :return: None
        """
        leaf_id = self.data_index + self.capacity - 1  # new leaf id
        self.data[self.data_index] = data  # save new node data to data array
        self.set_priority(leaf_id, priority)  # set priority in the tree
        self.data_index += 1  # move to the next index
        if self.data_index >= self.capacity:  # if full, override from beginning, but keep permanent data
            self.full = True
            self.data_index = self.data_index % self.capacity + self.permanent_size

    def get_data(self, priority):
        """
        Get data with priority
        :param priority: priority value to search for
        :return: node id, node priority, node data
        """
        parent_id = 0  # start from the top of the tree
        while True:
            left_child_id = 2 * parent_id + 1  # left child id=2*i+1
            right_child_id = left_child_id + 1  # left child id=2*i+2
            if left_child_id >= len(self.priority_tree):  # reached bottom of tree
                leaf_id = parent_id  # leaf found
                break
            if priority <= self.priority_tree[left_child_id]:  # priority smaller than left, search on left
                parent_id = left_child_id
            else:
                priority -= self.priority_tree[left_child_id]  # TODO: search on right
                parent_id = right_child_id

        data_id = leaf_id - self.capacity + 1
        return leaf_id, self.priority_tree[leaf_id], self.data[data_id]

    def set_priority(self, leaf_id, priority):
        """
        set priority of a node in a tree
        :param leaf_id: id in the tree
        :param priority: new priority
        :return: None
        """
        priority_change = priority - self.priority_tree[leaf_id]  # priority change
        self.priority_tree[leaf_id] = priority  # update priority of node
        while leaf_id != 0:  # loop to change all parent node priority as well
            leaf_id = (leaf_id - 1) // 2  # parent node id
            self.priority_tree[leaf_id] += priority_change  # increase parent node priorities as well

    @property
    def total_priority(self):
        return self.priority_tree[0]

    @property
    def priorities(self):
        return self.priority_tree[-self.capacity:]

    @property
    def data_size(self):
        return self.data[0].size


class Memory:
    def __init__(self, capacity, permanent_size=0):
        """
        Replay memory class with priority
        :param capacity: total capacity
        :param permanent_size: fixed data size which are not overridden
        """
        self.epsilon = 0.001  # small amount to avoid zero priority, Prioritized replay constants
        self.demo_epsilon = 1.0  # Demonstration priority bonus
        self.alpha = 0.4  # Prioritized replay exponent α, [0~1] convert the importance of TD error to priority

        self.beta = 0.6  # Prioritized replay importance sampling exponent
        self.beta_increment_per_sampling = 0.001

        self.abs_err_upper = 1.  # clipped abs error

        self.permanent_size = permanent_size
        self.sum_tree = SumTree(capacity, permanent_size)

    def __len__(self):
        return len(self.sum_tree)

    @property
    def full(self):
        return self.sum_tree.full

    def push(self, transition):
        """
        add transition to memory
        :param transition:
        :return:
        """
        # store transition with max priority already stored
        # for first transition, store with priority=abs_err_upper
        max_p = np.max(self.sum_tree.priorities)
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sum_tree.append(max_p, transition)  # set the max_p for new transition

    def sample(self, batch_size):
        """
        Randomly choose batch_size samples from memory, based on priorities
        :param batch_size:
        :return:
        """
        if not self.full:
            raise RuntimeError("Sampling should not execute when not full with capacity %d" % self.sum_tree.capacity)

        # sampled data
        batch_id = np.empty((batch_size,), dtype=np.int32)
        batch_data = np.empty((batch_size, self.sum_tree.data_size), dtype=object)
        batch_weight = np.empty((batch_size, 1))

        # divide priority into batches
        pri_seg = self.sum_tree.total_priority / batch_size
        # update beta (beta0->1)
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        # minimum priority
        min_prob = np.min(self.sum_tree.priorities) / self.sum_tree.total_priority
        assert min_prob > 0

        for i in range(batch_size):
            v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
            batch_id[i], priority, batch_data[i] = self.sum_tree.get_data(v)  # note: id is the index in sum tree
            prob = priority / self.sum_tree.total_priority  # P(i)
            batch_weight[i, 0] = np.power(prob / min_prob, -self.beta)
        return batch_id, batch_data, batch_weight

    def batch_update(self, tree_ids, abs_errors):
        """
        convert the importance of TD error to priority
        :param tree_ids:
        :param abs_errors: abs TD error
        :return:
        """
        abs_errors[self.permanent_size:] += self.epsilon
        # priorities of demo transitions are given a bonus of demo_epsilon, to boost the frequency that they are sampled
        abs_errors[:self.sum_tree.permanent_size] += self.demo_epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for t, p in zip(tree_ids, ps):
            self.sum_tree.set_priority(t, p)
