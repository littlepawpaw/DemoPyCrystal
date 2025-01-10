import numpy as np
import pyvista as pv

class UnitCell:
    def __init__(self,
                 a=1.0, b=1.0, c=1.0,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 r=None):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = np.deg2rad(alpha)
        self.beta = np.deg2rad(beta)
        self.gamma = np.deg2rad(gamma)
        self.r = r
        self.lattice = np.array([
            [i, j, k]
            for k in [0, 1]
            for j in [0, 1]
            for i in [0, 1]
        ])
        self.edge = np.array([[0, 1], [0, 2], [1, 3], [2, 3],
                              [0, 4], [1, 5], [2, 6], [3, 7],
                              [4, 5], [4, 6], [5, 7], [6, 7]])
        factor = (np.cos(self.alpha) - np.cos(self.beta) * np.cos(self.gamma)) / np.sin(self.gamma)
        self.transform = np.array([[1, np.cos(self.gamma), np.cos(self.beta)],
                                   [0, np.sin(self.gamma), factor],
                                   [0, 0, np.sqrt(1 - np.cos(self.beta) ** 2 - factor ** 2)]])
        self.matrix = np.matmul(self.transform,
                                np.array([[self.a, 0, 0],
                                          [0, self.b, 0],
                                          [0, 0, self.c]]))
        self.primitive = self.matrix

    def add_lattice(self, points: np.ndarray):
        self.lattice = np.append(self.lattice, points, axis=0)


    def cartesian(self):
        return np.matmul(self.matrix, self.lattice.transpose()).transpose()


class SC(UnitCell):
    def __init__(self, a=1.0):
        super().__init__(a=a, b=a, c=a)

        self.primitive = self.matrix

class BCC(UnitCell):
    def __init__(self, a=1.0):
        super().__init__(a=a, b=a, c=a)

        self.add_lattice(np.array([[0.5, 0.5, 0.5]]))

        bravais = np.array([
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ])
        self.primitive = np.matmul(bravais, self.matrix)

class FCC(UnitCell):
    def __init__(self, a=1.0):
        super().__init__(a=a, b=a, c=a)

        self.add_lattice(np.array([[0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0],
                                   [1, 0.5, 0.5],
                                   [0.5, 1, 0.5],
                                   [0.5, 0.5, 1]]))

        bravais = np.array([
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ])
        self.primitive = np.matmul(bravais, self.matrix)

class ST(UnitCell):
    def __init__(self, a=1.0, c=2):
        super().__init__(a=a, b=a, c=c)

        self.primitive_vectors = self.matrix


class BCT(UnitCell):
    def __init__(self, a=1.0, c=2):
        super().__init__(a=a, b=a, c=c)

        self.add_lattice(np.array([[0.5, 0.5, 0.5]]))

        bravais = np.array([
            [0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5]
        ])
        self.primitive = np.matmul(bravais, self.matrix)


class SO(UnitCell):
    def __init__(self, a=1.0, b=2, c=3):
        super().__init__(a=a, b=b, c=c)

        self.primitive_vectors = self.matrix


class BCO(UnitCell):
    def __init__(self, a=1.0, b=2, c=3):
        super().__init__(a=a, b=b, c=c)

        self.add_lattice(np.array([[0.5, 0.5, 0.5]]))

        bravais = np.array([
            [0.5, 0.5, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        self.primitive = np.matmul(bravais, self.matrix)


class FCO(UnitCell):
    def __init__(self, a=1.0, b=2, c=3):
        super().__init__(a=a, b=b, c=c)

        self.add_lattice(np.array([[0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0],
                                   [1, 0.5, 0.5],
                                   [0.5, 1, 0.5],
                                   [0.5, 0.5, 1]]))

        bravais = np.array([
            [0.5, 0.5, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5]
        ])
        self.primitive = np.matmul(bravais, self.matrix)


class SH(UnitCell):
    def __init__(self, a, c):
        super().__init__(a=a, b=a, c=c, gamma=120)

        bravais = np.array([
            [1, 0, 0],
            [-0.5, np.sqrt(3) / 2, 0],
            [0, 0, 1]
        ])

        self.primitive = np.matmul(bravais, self.matrix)