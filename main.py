from itertools import combinations
import numpy as np
import scipy
import pyvista as pv


class Cell:
    def __init__(self,
                 a=1, b=1, c=1,
                 alpha=90, beta=90, gamma=90,
                 r=None):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = np.deg2rad(alpha)
        self.beta = np.deg2rad(beta)
        self.gamma = np.deg2rad(gamma)
        self.r = r
        self.lattice = np.array([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [1, 1, 0],
                                 [0, 0, 1],
                                 [1, 0, 1],
                                 [0, 1, 1],
                                 [1, 1, 1]])
        # self.lattice = np.array([
        #     [i, j, k]
        #     for k in [0, 1]
        #     for j in [0, 1]
        #     for i in [0, 1]
        # ])
        self.edge = np.array([[0, 1], [0, 2], [1, 3], [2, 3],
                              [0, 4], [1, 5], [2, 6], [3, 7],
                              [4, 5], [4, 6], [5, 7], [6, 7]])
        factor = (np.cos(self.alpha) - np.cos(self.beta)*np.cos(self.gamma)) / np.sin(self.gamma)
        self.transform = np.array([[1, np.cos(self.gamma), np.cos(self.beta)],
                                   [0, np.sin(self.gamma), factor],
                                   [0, 0, np.sqrt(1 - np.cos(self.beta) ** 2 - factor ** 2)]])
        self.matrix = np.matmul(self.transform,
                                np.array([[self.a, 0, 0],
                                          [0, self.b, 0],
                                          [0, 0, self.c]]))
        # self.matrix = np.float16(self.matrix)
        self.plotter = pv.Plotter()

    def site(self, point: np.ndarray):
        self.lattice = np.append(self.lattice, point, axis=0)

    def cartesian(self):
        return np.matmul(self.matrix, self.lattice.transpose()).transpose()

    def plot(self, r=0.05, repeat=(0, 0, 0), ws=False):
        # Create a PyVista plotter
        plotter = self.plotter
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

        # Plot the translated unit cells (if translations are provided)
        for i in range(repeat[0] + 1):
            for j in range(repeat[1] + 1):
                for k in range(repeat[2] + 1):
                    if i or j or k:
                        # Calculate the translation vector
                        translation = np.matmul(self.matrix, np.array([i, j, k]))

                        # Translate the original lattice points
                        translated_points = lattice_points + translation

                        # Plot the translated points as spheres
                        for point in translated_points:
                            sphere = pv.Sphere(radius=0.05, center=point)
                            plotter.add_mesh(sphere, color='blue', smooth_shading=True)
                    else:
                        pass

        if ws:
            self.ws_cell()


    def ws_cell(self):
        plotter = self.plotter
        lattice_points = self.cartesian()

        # Include neighbors of the lattice points (translations in a 3x3x3 grid)
        neighbor_shifts = np.array([
            [i, j, k]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for k in [-1, 0, 1]
        ])
        neighbor_points = []
        for shift in neighbor_shifts:
            neighbor_points.extend(lattice_points + np.matmul(self.matrix, shift))
        neighbor_points = np.array(neighbor_points)

        # Create a Voronoi diagram around the origin
        vor = scipy.spatial.Voronoi(neighbor_points)

        # Find the Voronoi region corresponding to the origin (closest to [0, 0, 0])
        origin_index = np.argmin(np.linalg.norm(neighbor_points, axis=1))  # Closest to origin
        ws_region_idx = vor.point_region[origin_index]
        ws_region = vor.regions[ws_region_idx]

        # # Filter out unbounded regions (-1)
        # if -1 in ws_region:
        #     print("The Wigner-Seitz cell contains unbounded regions. Please adjust the lattice range.")
        #     return

        # Get the vertices corresponding to the WS cell region
        ws_cell_vertices = vor.vertices[ws_region]

        # Create the convex hull for the WS cell
        hull = scipy.spatial.ConvexHull(ws_cell_vertices)

        # Create PyVista PolyData for the WS cell
        ws_cell = pv.PolyData()
        ws_cell.points = ws_cell_vertices

        # Define the faces from the convex hull simplices
        faces = []
        for simplex in hull.simplices:
            faces.append(len(simplex))
            faces.extend(simplex)
        ws_cell.faces = faces

        # Add the WS cell to the plotter
        plotter.add_mesh(ws_cell, color='green', opacity=0.5, show_edges=False)

    def plane(self, h: int, k: int, l: int, size=1.0):
        # Reciprocal lattice vector
        g = np.array([h / self.a, k / self.b, l / self.c])

        # Plane normal in Cartesian coordinates
        normal = np.matmul(self.matrix.T, g)

        # Create a plane centered at the origin with the given normal
        plane = pv.Plane(
            center=[h, k, l],
            direction=normal,
            i_size=size * self.a,
            j_size=size * self.b
        )

        # Add the plane to the plotter
        self.plotter.add_mesh(plane, opacity=0.8)
    

    def clear(self):
        self.plotter = pv.Plotter()

    def show(self):
        self.plotter.show()


class SC(Cell):
    def __init__(self, a=1, **kwargs):
        super().__init__(a=a, b=a, c=a, **kwargs)


class BCC(Cell):
    def __init__(self, a=1, **kwargs):
        super().__init__(a=a, b=a, c=a, **kwargs)
        self.site(np.array([[0.5, 0.5, 0.5]]))


class FCC(Cell):
    def __init__(self, a=1, **kwargs):
        super().__init__(a=a, b=a, c=a, **kwargs)
        self.site(np.array([[0, 0.5, 0.5],
                            [0.5, 0, 0.5],
                            [0.5, 0.5, 0],
                            [1, 0.5, 0.5],
                            [0.5, 1, 0.5],
                            [0.5, 0.5, 1]]))


# sc = SC()
# sc.plot(repeat=(1, 1, 1), ws=True)
#
bcc = BCC()
# bcc.plot(repeat=(1,1,1), ws=True)
# bcc.plane(0,0,1, size=5)
# bcc.show()
#
# fcc = FCC()
# fcc.plot(repeat=(1,1,1), ws=True)
# fcc.plane(1,1,1, size=2)
# fcc.show()
# fcc.clear()





# cell = Cell(alpha=80, beta=80, gamma=80)
# cell.plot(repeat=(2, 2, 2), ws=True)

