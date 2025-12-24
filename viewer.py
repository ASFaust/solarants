from collections import deque
import time
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame


class Viewer:
    def __init__(
        self,
        system,
        control_handle,
        width: int = 1200,
        height: int = 1200,
        initial_zoom: float = 0.25,
        zoom_min: float = 1e-6,
        zoom_max: float = 1e3,
        zoom_step: float = 1.1,
        substeps_per_frame: int = 1,
        trail_min_delta: float = 2.0,
        trail_max_length: float = 20000.0,
        window_name: str = "Solar System Viewer",
    ):
        self.system = system
        self.width = int(width)
        self.height = int(height)
        self.window_name = window_name
        self.control_handle = control_handle

        self.center = np.array([0.0, 0.0], dtype=np.float64)
        self.zoom = float(initial_zoom)
        self.zoom_min = float(zoom_min)
        self.zoom_max = float(zoom_max)
        self.zoom_step = float(zoom_step)
        self.substeps_per_frame = int(substeps_per_frame)
        self.trail_min_delta = float(trail_min_delta)
        self.trail_max_length = float(trail_max_length)

        self.focus_name: Optional[str] = None
        self.hover_name: Optional[str] = None
        self.paused = False

        self._mouse_pos = (0, 0)
        self._last_bodies: List[object] = []
        self._last_body_by_name: Dict[str, object] = {}
        self._last_names: List[str] = []
        self._last_resources: List[object] = []

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
        self._ema_steps_per_sec: Optional[float] = None
        self._perf_last_time: Optional[float] = None
        self._perf_last_sim_steps = 0
        self._ema_alpha = 0.15
        self._show_gravity = True

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
        if not self._last_body_by_name:
            return None
        cursor_sim = self.screen_to_sim(screen_pos)
        best_name = None
        best_dist2 = None
        for name in self._last_names:
            body = self._last_body_by_name[name]
            dx = body.position[0] - cursor_sim[0]
            dy = body.position[1] - cursor_sim[1]
            dist = float(np.hypot(dx, dy)) - float(body.radius)
            if best_dist2 is None or dist < best_dist2:
                best_dist2 = dist
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
        def fmt(value: float, decimals: int = 3) -> str:
            return f"{value:,.{decimals}f}"

        if self._font is None:
            return

        entries: List[Tuple[str, str]] = [
            ("zoom", f"{self.zoom:.6f}"),
            ("paused", str(self.paused)),
            ("simulation speed", f"{self.substeps_per_frame}x"),
            ("simulation steps", f"{self._sim_steps:,}"),
        ]
        if self._ema_steps_per_sec is None:
            entries.append(("sim steps/sec (ema)", "n/a"))
        else:
            entries.append(("sim steps/sec (ema)", fmt(self._ema_steps_per_sec, 1)))
        if self.focus_name is not None:
            entries.append(("focus", self.focus_name))
        if self.hover_name is not None:
            entries.append(("hover", self.hover_name))
        if self.focus_name is not None and self.focus_name in self._last_body_by_name:
            body = self._last_body_by_name[self.focus_name]
            vx = body.velocity[0]
            vy = body.velocity[1]
            speed = float(np.hypot(vx, vy))
            entries.append(("vel", f"({fmt(vx)}, {fmt(vy)}) |v| {fmt(speed)}"))
            entries.append(("mass", fmt(body.mass)))
            entries.append(("density", fmt(body.density)))
            entries.append(("radius", fmt(body.radius)))
            entries.append(("surface gravity", fmt(body.surfaceGravity)))

        label_widths = [self._font.size(f"{label}:")[0] for label, _ in entries]
        max_label_width = max(label_widths, default=0)
        x_label = 10
        x_value = x_label + max_label_width + 8
        y = 8
        for label, value in entries:
            label_surf = self._font.render(f"{label}:", True, (220, 220, 220))
            value_surf = self._font.render(value, True, (220, 220, 220))
            surface.blit(label_surf, (x_label, y))
            surface.blit(value_surf, (x_value, y))
            y += 18

        controls = "LMB drag pan  LMB click focus  RMB drag ruler  wheel zoom  space pause  esc quit"
        label = self._font.render(controls, True, (180, 180, 180))
        surface.blit(label, (10, self.height - 20))

    def _update_camera_follow(self):
        if self.focus_name is None or not self._last_body_by_name:
            return
        if self.focus_name not in self._last_body_by_name:
            self._set_focus(None)
            return
        body = self._last_body_by_name[self.focus_name]
        target = np.array([body.position[0], body.position[1]], dtype=np.float64)
        self.center[:] = target

    def _draw_bodies(self, surface: pygame.Surface):
        for name in self._last_names:
            body = self._last_body_by_name[name]
            pos = np.array([body.position[0], body.position[1]], dtype=np.float64)
            px = self.sim_to_screen(pos)
            radius_px = max(2, int(round(body.radius * self.zoom)))
            color = (60, 200, 60)
            if name == self.hover_name:
                color = (0, 255, 255)
            pygame.draw.circle(surface, color, px, radius_px)
            if name == self.focus_name:
                pygame.draw.circle(surface, (255, 200, 50), px, max(radius_px + 3, 6), 2)

    def _draw_resources(self, surface: pygame.Surface):
        for resource in self._last_resources:
            pos = np.array([resource.position[0], resource.position[1]], dtype=np.float64)
            px = self.sim_to_screen(pos)
            radius_px = max(2, int(round(resource.radius * self.zoom)))
            pygame.draw.circle(surface, (220, 160, 60), px, radius_px)

    def _update_trails(self):
        min_delta2 = self.trail_min_delta * self.trail_min_delta
        origin = None
        if self.focus_name is not None and self.focus_name in self._last_body_by_name:
            focus_body = self._last_body_by_name[self.focus_name]
            origin = np.array([focus_body.position[0], focus_body.position[1]], dtype=np.float64)

        for name in self._last_names:
            body = self._last_body_by_name[name]
            pos = np.array([body.position[0], body.position[1]], dtype=np.float64)
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
        use_relative = self.focus_name is not None and self.focus_name in self._last_body_by_name
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

    def _draw_hover_gravity(self, surface: pygame.Surface):
        if self._font is None or self.hover_name is None or not self._show_gravity:
            return
        cursor_sim = self.screen_to_sim(self._mouse_pos)
        gravity = np.array(self.system.calculateGravity(cursor_sim), dtype=np.float64)
        magnitude = float(np.hypot(gravity[0], gravity[1]))
        if magnitude <= 0.0:
            return
        direction = gravity / magnitude
        start = np.array(self._mouse_pos, dtype=np.float64)
        arrow_len_px = 75.0
        end = start + direction * arrow_len_px
        color = (255, 140, 80)
        pygame.draw.line(surface, color, start, end, 2)
        head_len = 8.0
        head_width = 5.0
        perp = np.array([-direction[1], direction[0]], dtype=np.float64)
        tip = end
        left = tip - direction * head_len + perp * head_width
        right = tip - direction * head_len - perp * head_width
        pygame.draw.polygon(surface, color, [tip, left, right])
        label = self._font.render(f"{magnitude:,.3f}", True, color)
        label_pos = (int(end[0] + 6), int(end[1] - 10))
        surface.blit(label, label_pos)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption(self.window_name)
        self._font = pygame.font.Font(None, 18)
        clock = pygame.time.Clock()
        self._perf_last_time = time.perf_counter()
        self._perf_last_sim_steps = self._sim_steps

        while not self._should_quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._should_quit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._set_focus(None)
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_g:
                        self._show_gravity = not self._show_gravity
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
                            picked = self._pick_body(event.pos)
                            if picked == self.focus_name:
                                picked = None
                            self._set_focus(picked)
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
                elif event.type == pygame.VIDEORESIZE:
                    self.width = int(event.w)
                    self.height = int(event.h)
                    screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)

            if not self.paused:
                for _ in range(self.substeps_per_frame):
                    self.control_handle() #call the controlling function to set control forces
                    self.system.step(1)
                self._sim_steps += self.substeps_per_frame
            now = time.perf_counter()
            if self.paused:
                self._perf_last_time = now
                self._perf_last_sim_steps = self._sim_steps
            elif self._perf_last_time is not None:
                dt = now - self._perf_last_time
                if dt > 0.0:
                    sim_steps_delta = self._sim_steps - self._perf_last_sim_steps
                    inst_rate = sim_steps_delta / dt
                    if self._ema_steps_per_sec is None:
                        self._ema_steps_per_sec = inst_rate
                    else:
                        self._ema_steps_per_sec = (
                            (1.0 - self._ema_alpha) * self._ema_steps_per_sec
                            + self._ema_alpha * inst_rate
                        )
                self._perf_last_time = now
                self._perf_last_sim_steps = self._sim_steps

            self._last_bodies = list(self.system.bodies)
            self._last_body_by_name = {body.name: body for body in self._last_bodies}
            self._last_names = sorted(self._last_body_by_name.keys())
            self._last_resources = list(self.system.resources)
            self._mouse_pos = pygame.mouse.get_pos()
            self._update_hover()
            self._update_camera_follow()
            self._update_trails()

            screen.fill((0, 0, 0))
            self._draw_trails(screen)
            self._draw_bodies(screen)
            self._draw_resources(screen)
            self._draw_hover_gravity(screen)
            self._draw_ruler(screen)
            self._draw_hud(screen)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
