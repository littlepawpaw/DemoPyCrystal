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

    def reciprocal(self):
        return 2 * np.pi * np.linalg.inv(self.primitive).T

    def plot(self, r=0.05, primitive=False):
        # Create a PyVista plotter
        plotter = pv.Plotter()
        lattice_points = self.cartesian()

        # Plot the lattice points as spheres
        for point in self.cartesian():
            sphere = pv.Sphere(radius=r, center=point)
            plotter.add_mesh(sphere, color='blue', smooth_shading=True)

        # Add the edges to represent the cell boundaries
        cell_edges = self.edge
        lines = []
        for edge in cell_edges:
            lines.append([lattice_points[edge[0]], lattice_points[edge[1]]])

        # Convert to PolyData format for lines
        lines = np.array(lines)
        plotter.add_lines(lines.reshape(-1, 3), color="black")

        origin = np.array([0, 0, 0])

        if primitive:
            # Add the vectors
            for vec in self.primitive.T:
                plotter.add_arrows(cent=origin, direction=vec, color="red", smooth_shading=True)

        plotter.show()


"""
Cubic systems
"""


def cubic(group="P", **kwargs):
    if group == "P":
        return SC(**kwargs)
    elif group == "I":
        return BCC(**kwargs)
    elif group == "F":
        return FCC(**kwargs)
    else:
        raise ValueError("Wrong point group!")


class SC(UnitCell):
    def __init__(self, a=1.0, **kwargs):
        super().__init__(a=a, b=a, c=a, **kwargs)

        self.primitive = self.matrix


class BCC(UnitCell):
    def __init__(self, a=1.0, **kwargs):
        super().__init__(a=a, b=a, c=a, **kwargs)

        self.add_lattice(np.array([[0.5, 0.5, 0.5]]))

        bravais = np.array([
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


class FCC(UnitCell):
    def __init__(self, a=1.0, **kwargs):
        super().__init__(a=a, b=a, c=a, **kwargs)

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
        self.primitive = np.matmul(self.matrix, bravais)


"""
Tetragonal systems
"""


def tetragonal(group="P", **kwargs):
    if group == "P":
        return STetr(**kwargs)
    elif group == "I":
        return BCT(**kwargs)
    else:
        raise ValueError("Wrong point group!")


class STetr(UnitCell):
    def __init__(self, a=1.0, c=2, **kwargs):
        super().__init__(a=a, b=a, c=c, **kwargs)

        self.primitive_vectors = self.matrix


class BCT(UnitCell):
    def __init__(self, a=1.0, c=2, **kwargs):
        super().__init__(a=a, b=a, c=c, **kwargs)

        self.add_lattice(np.array([[0.5, 0.5, 0.5]]))

        bravais = np.array([
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


"""
Orthorhombic systems
"""


def orthorhombic(group="P", **kwargs):
    if group == "P":
        return SO(**kwargs)
    elif group == "I":
        return BCO(**kwargs)
    elif group == "C":
        return ECO(**kwargs)
    elif group == "F":
        return FCO(**kwargs)
    else:
        raise ValueError("Wrong point group!")


class SO(UnitCell):
    def __init__(self, a=1.0, b=2, c=3, **kwargs):
        super().__init__(a=a, b=b, c=c, **kwargs)

        self.primitive_vectors = self.matrix


class BCO(UnitCell):
    def __init__(self, a=1.0, b=2, c=3, **kwargs):
        super().__init__(a=a, b=b, c=c, **kwargs)

        self.add_lattice(np.array([[0.5, 0.5, 0.5]]))

        bravais = np.array([
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


class ECO(UnitCell):
    def __init__(self, a=1.0, b=2, c=3, **kwargs):
        super().__init__(a=a, b=b, c=c, **kwargs)

        self.add_lattice(np.array([
            [0.5, 0.5, 0],
            [0.5, 0.5, 1]
        ]))

        bravais = np.array([
            [0.5, 0.5, 0],
            [-0.5, 0.5, 0],
            [0, 0, 1]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


class FCO(UnitCell):
    def __init__(self, a=1.0, b=2, c=3, **kwargs):
        super().__init__(a=a, b=b, c=c, **kwargs)

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
        self.primitive = np.matmul(self.matrix, bravais)


"""
Monoclinic systems
"""


def monoclinic(group="P", **kwargs):
    if group == "P":
        return SM(**kwargs)
    elif group == "C":
        return ECM(**kwargs)
    else:
        raise ValueError("Wrong point group!")


class SM(UnitCell):
    def __init__(self, a=1.0, b=2, c=3, beta=120, **kwargs):
        super().__init__(a=a, b=b, c=c, beta=beta, **kwargs)

        bravais = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


class ECM(UnitCell):
    def __init__(self, a=1.0, b=2, c=3, beta=120, **kwargs):
        super().__init__(a=a, b=b, c=c, beta=beta, **kwargs)

        self.add_lattice(np.array([
            [0.5, 0.5, 0],
            [0.5, 0.5, 1]
        ]))

        bravais = np.array([
            [0.5, 0.5, 0],
            [-0.5, 0.5, 0],
            [0, 0, 1]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


"""
Triclinic systems
"""


def triclinic(group="P", **kwargs):
    if group == "P":
        return STric(**kwargs)
    else:
        raise ValueError("Wrong point group!")


class STric(UnitCell):
    def __init__(self, a=1.0, b=2, c=3, alpha=60, beta=70, gamma=80, **kwargs):
        super().__init__(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, **kwargs)

        bravais = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


"""
Trigonal systems
"""


def trigonal(group="P", **kwargs):
    if group == "P":
        return STrig(**kwargs)
    else:
        raise ValueError("Wrong point group!")


class STrig(UnitCell):
    def __init__(self, a=1.0, c=2, gamma=120, **kwargs):
        super().__init__(a=a, b=a, c=c, gamma=gamma, **kwargs)

        bravais = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


"""
Hexagonal systems
"""


def hexagonal(group="P", **kwargs):
    if group == "P":
        return SH(**kwargs)
    else:
        raise ValueError("Wrong point group!")


class SH(UnitCell):
    def __init__(self, a=1.0, c=2, gamma=120, **kwargs):
        super().__init__(a=a, b=a, c=c, gamma=gamma, **kwargs)

        bravais = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.primitive = np.matmul(self.matrix, bravais)


c1 = cubic("I")
c1.plot(primitive=True)
print(c1.reciprocal())
