"""Drawable shapes for the sketch canvas.

Every shape can draw itself and can also describe itself as pen strokes
(`to_strokes`), which is what the backend consumes. Coordinates are canvas
pixels.
"""

import math

import numpy as np
import pygame


class Shape:
    def __init__(self, color, weight):
        self.color = color
        self.weight = weight

    def display(self, surface):
        raise NotImplementedError

    def to_strokes(self):
        """Return a list of [N, 2] float arrays, one per pen stroke."""
        raise NotImplementedError

    def get_position(self):
        points = np.concatenate(self.to_strokes())
        return points.mean(axis=0)

    def get_length(self):
        raise NotImplementedError


class Polyline(Shape):
    """Freehand line through the recorded mouse positions."""

    def __init__(self, points, color, weight):
        super().__init__(color, weight)
        self.points = [tuple(p) for p in points]

    def add(self, point):
        if point != self.points[-1]:
            self.points.append(tuple(point))

    def display(self, surface):
        if len(self.points) == 1:
            pygame.draw.circle(surface, self.color, self.points[0], max(1, self.weight // 2))
            return
        pygame.draw.lines(surface, self.color, False, self.points, self.weight)

    def to_strokes(self):
        return [np.asarray(self.points, dtype=np.float64)]

    def get_length(self):
        (x0, y0), (x1, y1) = self.points[0], self.points[-1]
        return math.hypot(x1 - x0, y1 - y0)


class Rectangle(Shape):
    def __init__(self, x, y, width, height, color, weight):
        super().__init__(color, weight)
        # Normalize so that dragging in any direction gives a valid rectangle.
        self.x, self.y = min(x, x + width), min(y, y + height)
        self.width, self.height = abs(width), abs(height)

    def display(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height), self.weight)

    def to_strokes(self):
        x0, y0, x1, y1 = self.x, self.y, self.x + self.width, self.y + self.height
        return [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)]

    def get_position(self):
        return np.array([self.x + self.width / 2, self.y + self.height / 2])

    def get_length(self):
        return max(self.width, self.height)


class Circle(Shape):
    SEGMENTS = 64

    def __init__(self, center_x, center_y, radius, color, weight):
        super().__init__(color, weight)
        self.center_x, self.center_y, self.radius = center_x, center_y, radius

    def display(self, surface):
        pygame.draw.circle(surface, self.color, (self.center_x, self.center_y), self.radius, self.weight)

    def to_strokes(self):
        angles = np.linspace(0.0, 2.0 * np.pi, self.SEGMENTS + 1)
        return [np.stack([self.center_x + self.radius * np.cos(angles),
                          self.center_y + self.radius * np.sin(angles)], axis=1)]

    def get_position(self):
        return np.array([self.center_x, self.center_y], dtype=np.float64)

    def get_length(self):
        return self.radius * 2
