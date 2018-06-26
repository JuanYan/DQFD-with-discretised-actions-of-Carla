# -*- coding: utf-8 -*-
"""
RL replay memory
"""
import numpy as np

# set seed, so the results are repeatable
np.random.seed(0)


class SumTree:
    """
    A Sum Tree to store data with priority
    Leaf stores priority of data, and each parent node is the sum of child nodes
    """

    def __init__(self, capacity, permanent_data=0):
        """

        :param capacity:
        :param permanent_data:
        """
        self.full = False
        self.data_index = 0
        self.capacity = capacity
        self.priority_tree = np.zeros(2 * capacity - 1)  # stores priorities in leaf
        self.data = np.zeros(capacity, dtype=object)  # stores data
        self.permanent_data = permanent_data  # numbers of data which never be replaced TODO
        if not 0 <= self.permanent_data <= self.capacity:
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
            self.data_index = self.data_index % self.capacity + self.permanent_data

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

    def set_priority(self, node_id, priority):
        """
        set priority of a node in a tree
        :param node_id: id in the tree
        :param priority: new priority
        :return: None
        """
        priority_change = priority - self.priority_tree[node_id]  # priority change
        self.priority_tree[node_id] = priority  # update priority of node
        while node_id != 0:  # loop to change all parent node priority as well
            node_id = (node_id - 1) // 2  # parent node id
            self.priority_tree[node_id] += priority_change  # increase parent node priorities as well

    @property
    def total_priority(self):
        return self.priority_tree[0]


