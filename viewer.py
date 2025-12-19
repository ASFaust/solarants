from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame


class Viewer:
    def __init__(
        self,
        system,
        width: int = 1200,
        height: int = 1200,
        initial_zoom: float = 0.25,
        zoom_min: float = 1e-6,
        zoom_max: float = 1e3,
        zoom_step: float = 1.1,
        selection_radius_px: float = 6.0,
        substeps_per_frame: int = 1,
        trail_min_delta: float = 2.0,
        trail_max_length: float = 200.0,
        window_name: str = "Solar System Viewer",
    ):
        self.system = system
        self.width = int(width)
        self.height = int(height)
        self.window_name = window_name

        self.center = np.array([0.0, 0.0], dtype=np.float64)
        self.zoom = float(initial_zoom)
        self.zoom_min = float(zoom_min)
        self.zoom_max = float(zoom_max)
        self.zoom_step = float(zoom_step)
        self.selection_radius_px = float(selection_radius_px)
        self.substeps_per_frame = int(substeps_per_frame)
        self.trail_min_delta = float(trail_min_delta)
        self.trail_max_length = float(trail_max_length)

        self.focus_name: Optional[str] = None
        self.hover_name: Optional[str] = None
        self.paused = False

        self._mouse_pos = (0, 0)
        self._last_props: Dict[str, dict] = {}
        self._last_names: List[str] = []

        self._ruler_active = False
        self._ruler_start: Optional[np.ndarray] = None
        self._ruler_end: Optional[np.ndarray] = None
        self._should_quit = False
        self._font: Optional[pygame.font.Font] = None
        self._dragging = False
        self._drag_start_pos: Optional[Tuple[int, int]] = None
        self._drag_start_center: Optional[np.ndarray] = None
        self._drag_threshold_px = 4.0
        self._trails: Dict[str, Deque[np.ndarray]] = {}
        self._trail_lengths: Dict[str, float] = {}
        self._sim_steps = 0

    def sim_to_screen(self, sim_pos: np.ndarray) -> Tuple[int, int]:
        x = (sim_pos[0] - self.center[0]) * self.zoom + self.width * 0.5
        y = (sim_pos[1] - self.center[1]) * self.zoom + self.height * 0.5
        return int(round(x)), int(round(y))

    def _relative_to_screen(self, rel_pos: np.ndarray) -> Tuple[int, int]:
        x = rel_pos[0] * self.zoom + self.width * 0.5
        y = rel_pos[1] * self.zoom + self.height * 0.5
        return int(round(x)), int(round(y))

    def screen_to_sim(self, screen_pos: Tuple[int, int]) -> np.ndarray:
        x = (screen_pos[0] - self.width * 0.5) / self.zoom + self.center[0]
        y = (screen_pos[1] - self.height * 0.5) / self.zoom + self.center[1]
        return np.array([x, y], dtype=np.float64)

    def _pick_body(self, screen_pos: Tuple[int, int]) -> Optional[str]:
        if not self._last_props:
            return None
        cursor_sim = self.screen_to_sim(screen_pos)
        selection_radius_sim = self.selection_radius_px / max(self.zoom, 1e-12)
        best_name = None
        best_dist2 = None
        for name in self._last_names:
            prop = self._last_props[name]
            dx = prop["position_x"] - cursor_sim[0]
            dy = prop["position_y"] - cursor_sim[1]
            radius = prop["radius"] + selection_radius_sim
            dist2 = dx * dx + dy * dy
            if dist2 <= radius * radius:
                if best_dist2 is None or dist2 < best_dist2:
                    best_dist2 = dist2
                    best_name = name
        return best_name

    def _update_hover(self):
        self.hover_name = self._pick_body(self._mouse_pos)

    def _apply_zoom_at(self, screen_pos: Tuple[int, int], wheel_delta: float):
        if wheel_delta == 0.0:
            return
        scale = self.zoom_step ** wheel_delta
        new_zoom = float(np.clip(self.zoom * scale, self.zoom_min, self.zoom_max))
        if new_zoom == self.zoom:
            return
        anchor_sim = self.screen_to_sim(screen_pos)
        self.zoom = new_zoom
        self.center[0] = anchor_sim[0] - (screen_pos[0] - self.width * 0.5) / self.zoom
        self.center[1] = anchor_sim[1] - (screen_pos[1] - self.height * 0.5) / self.zoom

    def _clear_trails(self):
        self._trails = {}
        self._trail_lengths = {}

    def _set_focus(self, name: Optional[str]):
        if name != self.focus_name:
            self._clear_trails()
        self.focus_name = name

    def _draw_ruler(self, surface: pygame.Surface):
        if not (self._ruler_active and self._ruler_start is not None and self._ruler_end is not None):
            return

        start = self._ruler_start
        end = self._ruler_end
        start_px = self.sim_to_screen(start)
        end_px = self.sim_to_screen(end)

        pygame.draw.line(surface, (200, 200, 255), start_px, end_px, 1)
        pygame.draw.circle(surface, (200, 200, 255), start_px, 3)
        pygame.draw.circle(surface, (200, 200, 255), end_px, 3)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = float(np.hypot(dx, dy))
        text = f"dist: {dist:.3f}  dx: {dx:.3f}  dy: {dy:.3f}"
        if self._font is not None:
            label = self._font.render(text, True, (220, 220, 255))
            label_pos = (min(start_px[0], end_px[0]) + 6, min(start_px[1], end_px[1]) - 18)
            surface.blit(label, label_pos)

    def _draw_hud(self, surface: pygame.Surface):
        lines = [
            f"zoom: {self.zoom:.6f}",
            f"paused: {self.paused}",
            f"simulation speed: {self.substeps_per_frame}x",
            f"time units: {self._sim_steps}",
        ]
        if self.focus_name is not None:
            lines.append(f"focus: {self.focus_name}")
        if self.hover_name is not None:
            lines.append(f"hover: {self.hover_name}")
        if self.focus_name is not None and self.focus_name in self._last_props:
            prop = self._last_props[self.focus_name]
            vx = prop["velocity_x"]
            vy = prop["velocity_y"]
            speed = float(np.hypot(vx, vy))
            lines.append(f"vel: ({vx:.3f}, {vy:.3f}) |v| {speed:.3f}")

        if self._font is None:
            return

        y = 8
        for line in lines:
            label = self._font.render(line, True, (220, 220, 220))
            surface.blit(label, (10, y))
            y += 18

        controls = "LMB drag pan  LMB click focus  RMB drag ruler  wheel zoom  space pause  esc quit"
        label = self._font.render(controls, True, (180, 180, 180))
        surface.blit(label, (10, self.height - 20))

    def _update_camera_follow(self):
        if self.focus_name is None or not self._last_props:
            return
        if self.focus_name not in self._last_props:
            self._set_focus(None)
            return
        prop = self._last_props[self.focus_name]
        target = np.array([prop["position_x"], prop["position_y"]], dtype=np.float64)
        self.center[:] = target

    def _draw_bodies(self, surface: pygame.Surface):
        for name in self._last_names:
            prop = self._last_props[name]
            pos = np.array([prop["position_x"], prop["position_y"]], dtype=np.float64)
            px = self.sim_to_screen(pos)
            radius_px = max(2, int(round(prop["radius"] * self.zoom)))
            color = (60, 200, 60)
            if name == self.hover_name:
                color = (0, 255, 255)
            pygame.draw.circle(surface, color, px, radius_px)
            if name == self.focus_name:
                pygame.draw.circle(surface, (255, 200, 50), px, max(radius_px + 3, 6), 2)

    def _update_trails(self):
        min_delta2 = self.trail_min_delta * self.trail_min_delta
        origin = None
        if self.focus_name is not None and self.focus_name in self._last_props:
            focus_prop = self._last_props[self.focus_name]
            origin = np.array([focus_prop["position_x"], focus_prop["position_y"]], dtype=np.float64)

        for name in self._last_names:
            prop = self._last_props[name]
            pos = np.array([prop["position_x"], prop["position_y"]], dtype=np.float64)
            if origin is not None:
                pos = pos - origin
            if name not in self._trails:
                self._trails[name] = deque()
                self._trail_lengths[name] = 0.0
            trail = self._trails[name]
            trail_length = self._trail_lengths[name]
            if not trail:
                trail.append(pos)
                self._trail_lengths[name] = 0.0
                continue
            last = trail[-1]
            dx = pos[0] - last[0]
            dy = pos[1] - last[1]
            if (dx * dx + dy * dy) >= min_delta2:
                trail_length += float(np.hypot(dx, dy))
                trail.append(pos)
                while trail_length > self.trail_max_length and len(trail) > 1:
                    first = trail.popleft()
                    second = trail[0]
                    seg_dx = second[0] - first[0]
                    seg_dy = second[1] - first[1]
                    trail_length -= float(np.hypot(seg_dx, seg_dy))
                self._trail_lengths[name] = trail_length

        stale_names = set(self._trails.keys()) - set(self._last_names)
        for name in stale_names:
            self._trails.pop(name, None)
            self._trail_lengths.pop(name, None)

    def _draw_trails(self, surface: pygame.Surface):
        use_relative = self.focus_name is not None and self.focus_name in self._last_props
        for name in self._last_names:
            trail = self._trails.get(name)
            if not trail or len(trail) < 2:
                continue
            if use_relative:
                points = [self._relative_to_screen(p) for p in trail]
            else:
                points = [self.sim_to_screen(p) for p in trail]
            color = (80, 120, 180) if name != self.focus_name else (160, 200, 255)
            pygame.draw.lines(surface, color, False, points, 1)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.window_name)
        self._font = pygame.font.Font(None, 18)
        clock = pygame.time.Clock()

        while not self._should_quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._should_quit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._should_quit = True
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_PERIOD:
                        self.substeps_per_frame = max(1, self.substeps_per_frame * 2)
                    elif event.key == pygame.K_COMMA:
                        self.substeps_per_frame = max(1, self.substeps_per_frame // 2)
                elif event.type == pygame.MOUSEMOTION:
                    self._mouse_pos = event.pos
                    if event.buttons[0] and self._drag_start_pos is not None and self._drag_start_center is not None:
                        dx = event.pos[0] - self._drag_start_pos[0]
                        dy = event.pos[1] - self._drag_start_pos[1]
                        if not self._dragging and (dx * dx + dy * dy) >= self._drag_threshold_px ** 2:
                            self._dragging = True
                            self._set_focus(None)
                        if self._dragging:
                            self.center[0] = self._drag_start_center[0] - dx / self.zoom
                            self.center[1] = self._drag_start_center[1] - dy / self.zoom
                    if self._ruler_active and self._ruler_start is not None:
                        self._ruler_end = self.screen_to_sim(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self._mouse_pos = event.pos
                        self._dragging = False
                        self._drag_start_pos = event.pos
                        self._drag_start_center = self.center.copy()
                    elif event.button == 3:
                        self._mouse_pos = event.pos
                        self._ruler_active = True
                        self._ruler_start = self.screen_to_sim(event.pos)
                        self._ruler_end = self._ruler_start.copy()
                    elif event.button in (4, 5):
                        self._mouse_pos = event.pos
                        delta = 1 if event.button == 4 else -1
                        self._apply_zoom_at(self._mouse_pos, delta)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if not self._dragging:
                            self._set_focus(self._pick_body(event.pos))
                        self._dragging = False
                        self._drag_start_pos = None
                        self._drag_start_center = None
                    elif event.button == 3:
                        self._ruler_active = False
                        self._ruler_start = None
                        self._ruler_end = None
                elif event.type == pygame.MOUSEWHEEL:
                    self._mouse_pos = pygame.mouse.get_pos()
                    self._apply_zoom_at(self._mouse_pos, event.y)

            if not self.paused:
                for _ in range(self.substeps_per_frame):
                    self.system.step()
                self._sim_steps += self.substeps_per_frame

            self._last_props = self.system.getAllBodyProperties()
            self._last_names = sorted(self._last_props.keys())
            self._mouse_pos = pygame.mouse.get_pos()
            self._update_hover()
            self._update_camera_follow()
            self._update_trails()

            screen.fill((0, 0, 0))
            self._draw_trails(screen)
            self._draw_bodies(screen)
            self._draw_ruler(screen)
            self._draw_hud(screen)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
