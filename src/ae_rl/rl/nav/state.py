import numpy as np


class StateModel:
    """
    State representation for kitchen navigation.

    State consists of:
    1. 6x12 obstacle grid (72 values): 0=free, 1=obstacle
    2. Target info (3 values): bearing, distance, visible

    Total state size: 75

    Target info encoding:
    - bearing: [-1, 1] where 0=ahead, -1=left, +1=right
    - distance: [0, 1] where 0=at target, 1=far (10m+)
    - visible: 0 or 1
    """

    # State dimensions
    GRID_ROWS = 6
    GRID_COLS = 12
    GRID_SIZE = GRID_ROWS * GRID_COLS  # 72
    TARGET_INFO_SIZE = 3  # bearing, distance, visible
    TOTAL_STATE_SIZE = GRID_SIZE + TARGET_INFO_SIZE  # 75

    def __init__(self, numStates):
        self.gridMaxNum = numStates  # Number of rows (depth)
        self.segmap = None
        self.pathClass = 4
        self.arrSize = 12  # Number of columns (width)
        self.state = None

        # Grid cell size in pixels
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 12

        # Free space classes (things agent can walk on/through)
        self.FREE_SPACE_CLASSES = {'Floor', 'Rug', 'Carpet', 'Mat'}

        # Target info (updated by environment each step)
        self.target_bearing = 0.0
        self.target_distance = 1.0
        self.target_visible = 0.0

    def set_target_info(self, bearing, distance, visible):
        """
        Set target information for state representation.

        Args:
            bearing: normalized angle [-1, 1]
            distance: normalized distance [0, 1]
            visible: 1.0 if visible, 0.0 otherwise
        """
        self.target_bearing = bearing
        self.target_distance = distance
        self.target_visible = visible

    def resetState(self):
        """Reset state to empty grid (all free space)."""
        self.state = [[0 for j in range(self.arrSize)] for i in range(self.gridMaxNum)]
        self.target_bearing = 0.0
        self.target_distance = 1.0
        self.target_visible = 0.0
        return self.getFullState()

    def getState(self, segmap):
        """
        Convert segmentation map to occupancy grid state.

        Args:
            segmap: 2D array of object IDs from segmentation

        Returns:
            6x12 binary grid (0=free, 1=obstacle)
        """
        self.state = [[0 for j in range(self.arrSize)] for i in range(self.gridMaxNum)]

        # Agent position is at bottom center of segmentation image
        agtPosX = segmap.shape[1]
        agtPosY = segmap.shape[0] // 2
        xMax = self.gridMaxNum
        yMax = self.arrSize // 2

        for xBuc in range(self.gridMaxNum):
            gridX = agtPosX - ((xMax - xBuc) * self.GRID_HEIGHT)
            for yBuc in range(self.arrSize):
                gridY = (yBuc - yMax) * self.GRID_WIDTH + agtPosY

                # Bounds checking to prevent index errors
                if gridX < 0 or gridY < 0:
                    # Out of bounds - mark as obstacle (unknown = obstacle for safety)
                    self.state[xBuc][yBuc] = 1
                    continue

                gridX_end = min(gridX + self.GRID_WIDTH, segmap.shape[0])
                gridY_end = min(gridY + self.GRID_HEIGHT, segmap.shape[1])

                # Get the region of the segmentation map for this grid cell
                region = segmap[gridX:gridX_end, gridY:gridY_end]

                if region.size == 0:
                    # Empty region - mark as obstacle for safety
                    self.state[xBuc][yBuc] = 1
                    continue

                # Find the most common element in the region
                flattened = region.flatten()
                unique_elements, counts = np.unique(flattened, return_counts=True)
                most_common = unique_elements[np.argmax(counts)]

                # Mark as obstacle if not free space
                is_free = any(free_class in str(most_common) for free_class in self.FREE_SPACE_CLASSES)
                if not is_free:
                    self.state[xBuc][yBuc] = 1

        return self.state

    def getFullState(self):
        """
        Get complete state including target info.

        Returns:
            1D numpy array of size 75 (72 grid + 3 target info)
        """
        grid_flat = np.array(self.state).flatten()
        target_info = np.array([self.target_bearing, self.target_distance, self.target_visible])
        return np.concatenate([grid_flat, target_info])

    def getStateFlat(self, segmap):
        """Get state as flattened 1D array (for DDQN input)."""
        self.getState(segmap)
        return self.getFullState()

    def visualize_state(self, state=None):
        """
        Return ASCII visualization of the state grid.

        Useful for debugging. Shows:
        - '.' = free space
        - '#' = obstacle
        - 'A' = agent position (bottom center)
        """
        if state is None:
            state = self.state

        if len(np.array(state).flatten()) > self.GRID_SIZE:
            # State includes target info, extract just the grid
            state = np.array(state).flatten()[:self.GRID_SIZE]

        state_2d = np.reshape(state, (self.gridMaxNum, self.arrSize))

        lines = []
        for row_idx, row in enumerate(state_2d):
            line = ""
            for col_idx, cell in enumerate(row):
                if row_idx == self.gridMaxNum - 1 and col_idx == self.arrSize // 2:
                    line += "A"  # Agent position
                elif cell == 0:
                    line += "."
                else:
                    line += "#"
            lines.append(line)

        # Add target info
        lines.append(f"Target: bearing={self.target_bearing:.2f}, dist={self.target_distance:.2f}, vis={self.target_visible}")

        return "\n".join(lines)

    def count_obstacles(self, state=None):
        """Count number of obstacle cells in state."""
        if state is None:
            state = self.state
        return sum(sum(row) for row in state)

    def count_free_space(self, state=None):
        """Count number of free cells in state."""
        if state is None:
            state = self.state
        total = self.gridMaxNum * self.arrSize
        return total - self.count_obstacles(state)

    def get_forward_clearance(self, state=None):
        """
        Get number of free cells directly ahead of agent.

        Useful for deciding if forward movement is safe.
        """
        if state is None:
            state = self.state

        state_2d = np.reshape(state, (self.gridMaxNum, self.arrSize))
        center_col = self.arrSize // 2

        clearance = 0
        for row_idx in range(self.gridMaxNum - 1, -1, -1):
            if state_2d[row_idx][center_col] == 0:
                clearance += 1
            else:
                break

        return clearance
