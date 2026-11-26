"""Sketch -> label frontend.

Draw on the canvas, press "Classify" (or Enter) and the sketch is preprocessed
into normalized stroke-3 points and sent to the backend. The right panel shows
the top-k labels and, when the backend returns them, the retrieved neighbors.

    python frontend/app.py --config configs/frontend.json

Keys: Enter classify, S save, D delete last shape, R start over.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pygame

from backend import build_request, make_backend
from preprocess import preprocess, stroke3_to_lines
from shapes import Circle, Polyline, Rectangle

WIDTH, HEIGHT = 1200, 750
TOOLBAR = pygame.Rect(0, 0, 125, HEIGHT)
CANVAS = pygame.Rect(125, 0, 875, HEIGHT)
PANEL = pygame.Rect(1000, 0, 200, HEIGHT)
WINDOWS = [pygame.Rect(1000, 150 * i, 200, 150) for i in range(4)]
CLASSIFY_BUTTON = pygame.Rect(1040, 625, 120, 36)
SAVE_BUTTON = pygame.Rect(1040, 685, 120, 36)

BLACK, WHITE, GRAY = (0, 0, 0), (255, 255, 255), (100, 100, 100)

PALETTE = [
    (183, 24, 24), (255, 13, 13), (240, 41, 157),
    (236, 41, 240), (169, 41, 240), (116, 41, 240),
    (24, 22, 240), (22, 194, 240), (22, 240, 187),
    (51, 229, 52), (225, 245, 79), (255, 244, 31),
    (255, 157, 52), (234, 210, 147), (131, 114, 70),
    (255, 255, 255), (175, 175, 175), (0, 0, 0),
]
# Swatch centers: 3 columns x 6 rows, 40 px apart.
SWATCHES = [(color, (20 + 40 * (i % 3), 20 + 40 * (i // 3))) for i, color in enumerate(PALETTE)]
# (stroke weight, icon center)
BRUSHES = [(3, (60, 290)), (6, (60, 340)), (10, (60, 390))]
TOOLS = {"circle": pygame.Rect(40, 540, 40, 40), "rect": pygame.Rect(40, 605, 40, 40),
         "line": pygame.Rect(40, 663, 40, 40)}


def draw_lines_fit(surface, lines, box, color=BLACK):
    """Draw polylines scaled to fit inside `box`, keeping the aspect ratio."""
    every = np.concatenate(lines)
    lower, upper = every.min(axis=0), every.max(axis=0)
    extent = max((upper - lower).max(), 1e-6)
    factor = min(box.width, box.height) / extent
    offset = np.array(box.center) - (lower + upper) / 2 * factor
    for line in lines:
        if len(line) > 1:
            pygame.draw.lines(surface, color, False, (line * factor + offset).tolist(), 2)


class App:
    def __init__(self, config):
        self.config = config
        self.backend = make_backend(config["backend"])
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sketch to label")
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 20)
        self.shapes = []
        self.current = None   # shape being dragged
        self.anchor = None    # where the drag started
        self.color = BLACK
        self.weight = BRUSHES[0][0]
        self.tool = "line"
        self.response = None
        self.request = None
        self.status = "Draw a sketch, then press Classify."

    # ----- sketch state -----

    def strokes(self):
        return [stroke for shape in self.shapes for stroke in shape.to_strokes()]

    def changed(self):
        """The canvas no longer matches the last answer."""
        self.request = self.response = None

    def classify(self):
        settings = self.config["preprocessing"]
        points = preprocess(self.strokes(), settings["scale"], settings["box"], settings["rdp_epsilon"])
        if len(points) == 0:
            self.status = "Nothing to classify yet."
            return
        self.request = build_request(points, self.config["top_k"])
        try:
            self.response = self.backend.query(self.request)
        except Exception as error:  # network or server failure: keep the UI alive
            self.response = None
            self.status = f"Backend error: {error}"
            return
        self.status = f"Sent {len(points)} points to the {self.backend.name} backend."
        if len(points) > settings["max_seq_len"]:
            self.status += f" Longer than the training limit ({settings['max_seq_len']})."

    def save(self):
        if self.request is None:
            self.classify()
        if self.request is None:
            return
        directory = Path(self.config["save_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"sketch_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps({"request": self.request, "response": self.response}), encoding="utf-8")
        self.status = f"Saved {path}"

    # ----- input -----

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d and self.shapes:
                self.shapes.pop()
                self.changed()
            elif event.key == pygame.K_r:
                self.shapes.clear()
                self.changed()
                self.status = "Cleared."
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.classify()
            elif event.key == pygame.K_s:
                self.save()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if CANVAS.collidepoint(event.pos):
                self.start_shape(event.pos)
            elif TOOLBAR.collidepoint(event.pos):
                self.select(event.pos)
            elif CLASSIFY_BUTTON.collidepoint(event.pos):
                self.classify()
            elif SAVE_BUTTON.collidepoint(event.pos):
                self.save()
        elif event.type == pygame.MOUSEMOTION and self.current is not None:
            self.update_shape(self.clamp(event.pos))
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.current is not None:
            if self.current.get_length() > 0 or isinstance(self.current, Polyline):
                self.shapes.append(self.current)
                self.changed()
            self.current = None

    @staticmethod
    def clamp(position):
        return (min(max(position[0], CANVAS.left), CANVAS.right - 1),
                min(max(position[1], CANVAS.top), CANVAS.bottom - 1))

    def select(self, position):
        for color, center in SWATCHES:
            if math.dist(position, center) <= 15:
                self.color = color
        for weight, center in BRUSHES:
            if math.dist(position, center) <= 20:
                self.weight = weight
        for tool, box in TOOLS.items():
            if box.collidepoint(position):
                self.tool = tool

    def start_shape(self, position):
        self.anchor = position
        if self.tool == "line":
            self.current = Polyline([position], self.color, self.weight)
        else:
            self.update_shape(position)

    def update_shape(self, position):
        (x0, y0), (x1, y1) = self.anchor, position
        if self.tool == "line":
            self.current.add(position)
        elif self.tool == "rect":
            self.current = Rectangle(x0, y0, x1 - x0, y1 - y0, self.color, self.weight)
        else:
            self.current = Circle(x0, y0, math.hypot(x1 - x0, y1 - y0), self.color, self.weight)

    # ----- drawing -----

    def draw(self):
        screen = self.screen
        screen.fill(WHITE)
        screen.set_clip(CANVAS)
        for shape in self.shapes + ([self.current] if self.current else []):
            shape.display(screen)
        screen.set_clip(None)
        self.draw_toolbar()
        self.draw_panel()
        screen.blit(self.small.render(self.status, True, BLACK), (CANVAS.left + 8, 8))
        hint = self.small.render("Enter classify; S save; D delete; R start over", True, GRAY)
        screen.blit(hint, (CANVAS.right - hint.get_width() - 8, HEIGHT - 22))
        pygame.display.flip()

    def draw_toolbar(self):
        screen = self.screen
        pygame.draw.rect(screen, WHITE, TOOLBAR)
        for top in (0, 250, 500):
            pygame.draw.rect(screen, BLACK, (0, top, TOOLBAR.width, 250), 1)
        for color, center in SWATCHES:
            pygame.draw.circle(screen, color, center, 12)
            pygame.draw.circle(screen, BLACK, center, 12, 2)
            if color == self.color:
                pygame.draw.rect(screen, BLACK, (center[0] - 16, center[1] - 16, 32, 32), 1)
        for weight, center in BRUSHES:
            pygame.draw.circle(screen, BLACK, center, weight)
            if weight == self.weight:
                pygame.draw.rect(screen, BLACK, (center[0] - 20, center[1] - 20, 40, 40), 1)
        pygame.draw.circle(screen, BLACK, TOOLS["circle"].center, 17, 2)
        pygame.draw.rect(screen, BLACK, TOOLS["rect"].inflate(-10, -10), 2)
        pygame.draw.lines(screen, BLACK, False, [(48, 688), (57, 678), (63, 688), (72, 678)], 3)
        pygame.draw.rect(screen, BLACK, TOOLS[self.tool].inflate(8, 8), 1)

    def draw_panel(self):
        screen = self.screen
        pygame.draw.rect(screen, WHITE, PANEL)
        for window in WINDOWS:
            pygame.draw.rect(screen, BLACK, window, 1)
        pygame.draw.rect(screen, (20, 20, 20), (PANEL.left, 600, PANEL.width, 150))
        for button, label in ((CLASSIFY_BUTTON, "Classify"), (SAVE_BUTTON, "Save")):
            pygame.draw.rect(screen, WHITE, button)
            text = self.font.render(label, True, BLACK)
            screen.blit(text, text.get_rect(center=button.center))

        title = self.font.render(f"Top {self.config['top_k']} ({self.backend.name})", True, BLACK)
        screen.blit(title, (WINDOWS[0].left + 10, 8))
        if self.response is None:
            return
        for row, prediction in enumerate(self.response.get("predictions", [])[:5]):
            line = f"{row + 1}. {prediction['label']}  {prediction['score']:.2f}"
            screen.blit(self.small.render(line, True, BLACK), (WINDOWS[0].left + 10, 36 + 22 * row))
        for window, neighbor in zip(WINDOWS[1:], self.response.get("neighbors", [])):
            caption = f"{neighbor['label']}  {neighbor['score']:.2f}"
            screen.blit(self.small.render(caption, True, BLACK), (window.left + 10, window.top + 6))
            lines = stroke3_to_lines(neighbor["points"])
            draw_lines_fit(screen, lines, pygame.Rect(window.left + 40, window.top + 28, 120, 112))

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                self.handle_event(event)
            self.draw()
            clock.tick(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "frontend.json")
    parser.add_argument("--backend-url", help="Use the HTTP backend at this URL instead of the configured one")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.backend_url:
        config["backend"].update(type="http", url=args.backend_url)
    pygame.display.init()
    pygame.font.init()
    App(config).run()
    pygame.quit()


if __name__ == "__main__":
    main()
