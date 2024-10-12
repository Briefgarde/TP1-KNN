import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self):
        self.colors = [
            "#EE82EE",
            "#FFD700",
            "#FF7F50",
            "#9ACD32",
            "#00CED1",
            "#9370DB",
            "#DEB887",
            "#FFA07A",
            "#ADFF2F",
            "#00FFFF",
        ]
        self.clear()

    @property
    def color(self):
        self.color_idx = (self.color_idx + 1) % len(self.colors)
        return self.colors[self.color_idx]

    def clear(self):
        try:
            del self.fig
            del self.ax
        except:
            pass

        self.color_idx = -1
        self.x_max = 0
        self.y_max = 0
        self.x_min = 0
        self.y_min = 0
        self.fig, self.ax = plt.subplots()

    def show(self):
        return self.fig

    def update_axis(self, vector):
        self.x_max = max(vector[0].max(), self.x_max)
        self.y_max = max(vector[1].max(), self.y_max)
        self.x_min = min(vector[0].min(), self.x_min)
        self.y_min = min(vector[1].min(), self.y_min)
        self.ax.set_xlim((-1 + self.x_min, self.x_max + 1))
        self.ax.set_ylim((-1 + self.y_min, self.y_max + 1))

    def set_label(self, origin, vector, label, fontsize=15):
        x_offset = -0.2 * len(label)
        y_offset = 0.2
        x = origin[0] + (vector[0] / 2) + x_offset
        y = origin[1] + (vector[1] / 2) + y_offset
        self.ax.text(x, y, label, fontsize=fontsize)

    def add_vector(self, vector, origin=None, label=None):
        vector = np.array(vector)
        if origin is None:
            origin = [0] * vector.shape[0]

        self.ax.quiver(
            *origin,
            vector[0],
            vector[1],
            color=self.color,
            scale_units="xy",
            angles="xy",
            scale=1,
        )
        if label is not None:
            self.set_label(origin, vector, label)

        self.update_axis(vector + np.array(origin))
