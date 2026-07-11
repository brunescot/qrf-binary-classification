import math

import warnings


class Leaves:

    def __init__(self, leaves_p):
        self.p = leaves_p.copy()
        self.alpha = dict()
        for j in leaves_p:
            self.alpha[j] = math.acos(math.sqrt(leaves_p[j]))


class TreeForQuantumPrediction:

    def __init__(self, h, attr_ind, leaves_proba):
        self.h = h
        self.n_nodes = 2 ** h - 1
        self.attr_ind = attr_ind.copy()
        self.leaves_proba = leaves_proba.copy()
        self.leaves = Leaves(leaves_proba)


class RandomForestForQuantumPrediction:

    def __init__(self, h, trees=None):  # , trees):
        self.h_trees = h
        if trees is None:
            self.trees = self.__test_trees()
        else:
            self.trees = trees.copy()
        self.n_trees = len(self.trees)
        self.angle_list = self.__angle_computing()

    def __angle_computing(self):
        rf_angle_list = []
        for tree in self.trees:
            for i in range(2 ** (self.h_trees - 1)):
                rf_angle_list.append(2 * tree.leaves.alpha[i])
        return rf_angle_list

    def __test_trees(self):
        tree1 = TreeForQuantumPrediction(
            3, [-1, 0, 1, 2], {0: 0.02, 1: 0.4, 2: 0.11, 3: 0.65}
        )
        tree2 = TreeForQuantumPrediction(
            3, [-1, 1, 0, 2], {0: 0.03, 1: 0.75, 2: 0.2, 3: 0.12}
        )
        tree3 = TreeForQuantumPrediction(
            3, [-1, 2, 0, 1], {0: 0.04, 1: 0.1, 2: 0.6, 3: 0.18}
        )
        tree4 = TreeForQuantumPrediction(
            3, [-1, 1, 2, 0], {0: 0.07, 1: 0.8, 2: 0.3, 3: 0.03}
        )
        return [tree1, tree2, tree3, tree4]

    def predict_proba(self, x):
        p = 0
        for tree in self.trees:
            node = 1
            for _ in range(tree.h - 1):
                if x[tree.attr_ind[node]] == '0':
                    node *= 2
                else:
                    node = node * 2 + 1
            node = node % (2 ** (tree.h - 1))
            p += tree.leaves_proba[node]
        return p / 4