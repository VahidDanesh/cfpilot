"""

Grid map library in python

author: Atsushi Sakai, some improvements by Vahid Danesh

"""
import matplotlib.pyplot as plt
import numpy as np


class GridMap:
    """
    GridMap class
    """

    def __init__(self, width: int, height: int, resolution: float,
                 center_x: float, center_y: float, init_val: float = 0.0):
        """__init__

        :param width: number of grid for width
        :param height: number of grid for height
        :param resolution: grid resolution [m]
        :param center_x: center x position  [m]
        :param center_y: center y position [m]
        :param init_val: initial value for all grid (float)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.center_x = center_x
        self.center_y = center_y

        self.left_lower_x = self.center_x - self.width / 2.0 * self.resolution
        self.left_lower_y = self.center_y - self.height / 2.0 * self.resolution

        self.n_data = self.width * self.height
        # Use NumPy array for fast operations (store floats directly)
        self.data = np.full((self.width, self.height), init_val, dtype=float)

    def valid(self, x_ind, y_ind) -> bool:
        """Check if indices are within grid bounds"""
        
        # output an array of bools
        x_ind = np.atleast_1d(x_ind)
        y_ind = np.atleast_1d(y_ind)
        return (0 <= x_ind) & (x_ind < self.width) & (0 <= y_ind) & (y_ind < self.height)
    
    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index

        when the index is out of grid map area, return None for that index

        :param x_ind: x index (int or array-like)
        :param y_ind: y index (int or array-like)
        :return: value(s) from grid, None for out of bounds indices
        """
        is_scalar = np.isscalar(x_ind) and np.isscalar(y_ind)
        x_ind = np.atleast_1d(x_ind)
        y_ind = np.atleast_1d(y_ind)
        
        valid_mask = self.valid(x_ind, y_ind)
        
        # Return None if any index is out of bounds
        if not np.all(valid_mask):
            return None
        
        result = self.data[x_ind, y_ind]
        return result.item() if is_scalar else result

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m] (float or array-like)
        :param y_pos: y position [m] (float or array-like)
        :return: tuple of (x_ind, y_ind) or (None, None) if out of bounds
        """
        x_pos = np.atleast_1d(x_pos)
        y_pos = np.atleast_1d(y_pos)
        
        x_ind = self.calc_xy_index_from_position(x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(y_pos, self.left_lower_y, self.height)

        return x_ind, y_ind
    
    def get_xy_poss_from_xy_indexes(self, x_ind, y_ind):
        """get_pos_from_xy_index

        :param x_ind: x index (int or array-like)
        :param y_ind: y index (int or array-like)
        :return: tuple of (x_pos, y_pos) or (None, None) if out of bounds
        """
        x_ind = np.atleast_1d(x_ind)
        y_ind = np.atleast_1d(y_ind)

        if np.all((0 <= x_ind) & (x_ind < self.width)) and np.all((0 <= y_ind) & (y_ind < self.height)):
            x_pos = self.left_lower_x + x_ind * self.resolution + self.resolution / 2.0
            y_pos = self.left_lower_y + y_ind * self.resolution + self.resolution / 2.0
            return x_pos, y_pos
        else:
            return None, None

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m] (float or array-like)
        :param y_pos: y position [m] (float or array-like)
        :param val: grid value (float or array-like)
        """
        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)

        if x_ind is None or y_ind is None:
            return False  # NG

        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index (int or array-like)
        :param y_ind: y index (int or array-like)
        :param val: grid value (float or array-like)
        """
        x_ind = np.atleast_1d(x_ind)
        y_ind = np.atleast_1d(y_ind)
        val = np.atleast_1d(val)
        
        if x_ind is None or y_ind is None:
            return False

        valid_mask = self.valid(x_ind, y_ind)
        
        if np.all(valid_mask):
            self.data[x_ind, y_ind] = val
            return True  # OK
        else:
            return False  # NG

    def set_value_from_polygon(self, pol_x, pol_y, val: float, inside: bool = True):
        """set_value_from_polygon

        Setting value inside or outside polygon

        :param pol_x: x position list for a polygon
        :param pol_y: y position list for a polygon
        :param val: grid value (float)
        :param inside: setting data inside or outside
        """
        # making ring polygon
        if (pol_x[0] != pol_x[-1]) or (pol_y[0] != pol_y[-1]):
            pol_x = np.append(pol_x, pol_x[0])
            pol_y = np.append(pol_y, pol_y[0])

        # setting value for all grid
        for x_ind in range(self.width):
            for y_ind in range(self.height):
                x_pos, y_pos = self.calc_grid_central_xy_position_from_xy_index(
                    x_ind, y_ind)

                flag = self.check_inside_polygon(x_pos, y_pos, pol_x, pol_y)

                if flag is inside:
                    self.set_value_from_xy_index(x_ind, y_ind, val)

    def add_obstacle(self, center_x: float, center_y: float, radius: float, 
                     val: float = 1.0, n_sides: int = 16):
        """Add a circular obstacle to the grid map
        
        The circle is approximated by a polygon with n_sides (default 16).
        
        :param center_x: x position of obstacle center [m]
        :param center_y: y position of obstacle center [m]
        :param radius: radius of the obstacle [m]
        :param val: grid value for the obstacle (default 1.0)
        :param n_sides: number of sides for polygon approximation (default 16)
        """
        # Generate polygon vertices approximating a circle
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        pol_x = center_x + radius * np.cos(angles)
        pol_y = center_y + radius * np.sin(angles)
        
        # Set values inside the polygon
        self.set_value_from_polygon(pol_x, pol_y, val, inside=True)

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        """Calculate flat grid index from x,y indices (for compatibility)"""
        x_ind = np.atleast_1d(x_ind)
        y_ind = np.atleast_1d(y_ind)
        grid_ind = (y_ind * self.width + x_ind).astype(int)
        return grid_ind.item() if grid_ind.size == 1 else grid_ind

    def calc_xy_index_from_grid_index(self, grid_ind):
        """Calculate x,y indices from flat grid index (for compatibility)"""
        grid_ind = np.atleast_1d(grid_ind)
        y_ind, x_ind = np.divmod(grid_ind, self.width)
        if grid_ind.size == 1:
            return x_ind.item(), y_ind.item()
        return x_ind, y_ind

    def calc_grid_index_from_xy_pos(self, x_pos, y_pos):
        """calc_grid_index_from_xy_pos

        :param x_pos: x position [m] (float or array-like)
        :param y_pos: y position [m] (float or array-like)
        """
        x_ind = self.calc_xy_index_from_position(x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(y_pos, self.left_lower_y, self.height)

        return self.calc_grid_index_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_grid_index(self, grid_ind):
        """Calculate central x,y position from flat grid index"""
        x_ind, y_ind = self.calc_xy_index_from_grid_index(grid_ind)
        return self.calc_grid_central_xy_position_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        """Calculate central x,y position from x,y indices"""
        x_ind = np.atleast_1d(x_ind)
        y_ind = np.atleast_1d(y_ind)
        
        x_pos = self.calc_grid_central_xy_position_from_index(x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(y_ind, self.left_lower_y)

        if x_pos.size == 1:
            return x_pos.item(), y_pos.item()
        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos: float):
        """Calculate central position from index"""
        index = np.atleast_1d(index)
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_xy_index_from_position(self, pos, lower_pos: float, max_index: int):
        """Calculate index from position"""
        pos = np.atleast_1d(pos)
        ind = np.floor((pos - lower_pos) / self.resolution).astype(int)
        valid_mask = (0 <= ind) & (ind <= max_index)
        
        if not np.all(valid_mask):
            return None
        
        return ind.item() if ind.size == 1 else ind

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val: float):
        """Check if grid cell is occupied based on threshold value"""
        val = self.get_value_from_xy_index(x_ind, y_ind)

        if val is None:
            return True
        
        val = np.atleast_1d(val)
        return val >= occupied_val

    def expand_grid(self, occupied_val: float = 1.0):
        """Expand occupied grid cells to neighboring cells"""
        x_inds, y_inds = np.where(self.data >= occupied_val)
        values = self.data[x_inds, y_inds]

        for (ix, iy, value) in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)

    def occupy_boundaries(self, boundary_width: int = 1, val: float = 1.0):
        """Occupy the boundary cells of the grid map using vectorized operations
        
        :param boundary_width: width of the boundary to occupy (in grid cells)
        :param val: value to set for the boundary cells
        """
        # Validate boundary_width
        max_boundary = min(self.width // 2, self.height // 2)
        boundary_width = min(boundary_width, max_boundary)
        
        # Top and bottom boundaries
        self.data[:boundary_width, :] = val  # Bottom
        self.data[-boundary_width:, :] = val  # Top
        
        # Left and right boundaries
        self.data[:, :boundary_width] = val  # Left
        self.data[:, -boundary_width:] = val  # Right


    def simplify_path(self, pathx, pathy, obstacle_threshold=0.5):
        """Remove unnecessary waypoints - keep only points where path must turn"""
        
        if len(pathx) <= 2:
            return pathx, pathy
        
        def line_clear(i, j):
            """Check if straight line between two points is obstacle-free"""
            n = int(np.hypot(pathx[j]-pathx[i], pathy[j]-pathy[i])) * 2 + 1
            xs = np.linspace(pathx[i], pathx[j], n).astype(int)
            ys = np.linspace(pathy[i], pathy[j], n).astype(int)
            return np.all(self.data[xs, ys] <= obstacle_threshold)
        
        result = [0]  # Keep start
        i = 0
        
        while i < len(pathx) - 1:
            # Skip ahead as far as possible
            j = len(pathx) - 1
            while j > i and not line_clear(i, j):
                j -= 1
            result.append(j)
            i = j
        
        return [pathx[k] for k in result], [pathy[k] for k in result]

    @staticmethod
    def check_inside_polygon(iox, ioy, x, y):

        n_point = len(x) - 1
        inside = False
        for i1 in range(n_point):
            i2 = (i1 + 1) % (n_point + 1)

            if x[i1] >= x[i2]:
                min_x, max_x = x[i2], x[i1]
            else:
                min_x, max_x = x[i1], x[i2]
            if not min_x <= iox < max_x:
                continue

            tmp1 = (y[i2] - y[i1]) / (x[i2] - x[i1])
            if (y[i1] + tmp1 * (iox - x[i1]) - ioy) > 0.0:
                inside = not inside

        return inside

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)
        print("n_data:", self.n_data)

    def plot_grid_map(self, ax=None, use_world_coords=False):
        """Plot the grid map as a heatmap
        
        :param ax: matplotlib axis to plot on (creates new if None)
        :param use_world_coords: if True, use real world coordinates; if False, use indices
        """
        grid_data = self.data.T # (data stored in (width, height) or (x, y) order, 
                                # so transpose for correct orientation for plot)
        if not ax:
            fig, ax = plt.subplots()
        
        if use_world_coords:
            # Calculate extent in world coordinates
            extent = [
                self.left_lower_x,
                self.left_lower_x + self.width * self.resolution,
                self.left_lower_y,
                self.left_lower_y + self.height * self.resolution
            ]
            heat_map = ax.imshow(grid_data, cmap="Blues", vmin=0.0, vmax=1.0,
                                origin='lower', extent=extent)
            ax.set_xlabel('X Position [m]')
            ax.set_ylabel('Y Position [m]')
            plt.axis("equal")
        else:
            heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xlabel('X Index')
            ax.set_ylabel('Y Index')
            plt.axis("equal")
        
        ax.set_title('Grid Map')

        return heat_map


def polygon_set_demo():
    """Demo function showing polygon-based grid value setting"""
    ox = [0.0, 4.35, 20.0, 50.0, 100.0, 130.0, 40.0]
    oy = [0.0, -4.15, -20.0, 0.0, 30.0, 60.0, 80.0]

    grid_map = GridMap(600, 290, 0.7, 60.0, 30.5)

    grid_map.set_value_from_polygon(ox, oy, 1.0, inside=False)

    grid_map.plot_grid_map()

    plt.axis("equal")
    plt.grid(True)


def position_set_demo():
    """Demo function showing position-based grid value setting"""
    grid_map = GridMap(100, 120, 0.5, 10.0, -0.5)

    grid_map.set_value_from_xy_pos(10.1, -1.1, 1.0)
    grid_map.set_value_from_xy_pos(10.1, -0.1, 1.0)
    grid_map.set_value_from_xy_pos(10.1, 1.1, 1.0)
    grid_map.set_value_from_xy_pos(11.1, 0.1, 1.0)
    grid_map.set_value_from_xy_pos(10.1, 0.1, 1.0)
    grid_map.set_value_from_xy_pos(9.1, 0.1, 1.0)

    grid_map.plot_grid_map()

    plt.axis("equal")
    plt.grid(True)


def main():
    print("start!!")

    position_set_demo()
    polygon_set_demo()

    plt.show()

    print("done!!")


if __name__ == '__main__':
    main()
