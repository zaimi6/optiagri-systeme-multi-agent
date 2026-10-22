from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
RIGHT_PANEL_WIDTH = 356
PANEL_WIDTH = RIGHT_PANEL_WIDTH
FPS = 48

MARGIN = 12
PANEL_GAP = 10
HEADER_HEIGHT = 48
BOTTOM_PANEL_HEIGHT = 156
BUTTON_HEIGHT = 32
BUTTON_WIDTH = 78
BUTTON_GAP = 8
PANEL_PADDING = 12
CARD_SPACING = 8
SIM_PANEL_PADDING = PANEL_PADDING
SIM_SECTION_GAP = CARD_SPACING
SIM_HEADER_INFO_HEIGHT = 76
SIM_FOOTER_INFO_HEIGHT = 60
GRAPH_SIDE_PADDING = 18
GRAPH_TOP_PADDING = 4
GRAPH_PANEL_MIN_HEIGHT = 120
STATUS_BOX_HEIGHT = 26
FLOW_HINT_HEIGHT = 18
SIDE_PANEL_PADDING = PANEL_PADDING
CHART_PANEL_PADDING = 12

FONT_SIZE_LABEL = 13
FONT_SIZE_SMALL = FONT_SIZE_LABEL
FONT_SIZE_MEDIUM = 16
FONT_SIZE_LARGE = 20
FONT_SIZE_TITLE = 24
FONT_SIZE_SUBTITLE = 15
FONT_SIZE_METRIC = 20
FONT_SIZE_VALUE = FONT_SIZE_METRIC

NODE_WIDTH = 108
NODE_HEIGHT = 42
AGENT_NODE_WIDTH = NODE_WIDTH
AGENT_NODE_HEIGHT = NODE_HEIGHT

DATA_LOAD_DELAY = 1.45
AGENT_ACTION_DELAY = 1.45
FUZZY_DELAY = 1.60
YIELD_UPDATE_DELAY = 1.75
OPTIMIZATION_DELAY = 1.80
SCENARIO_END_DELAY = 1.25
INTERACTION_PAUSE_DELAY = 0.40


class VisualizationAgent:
    """Render a compact, readable live window for the agricultural simulation."""

    def __init__(
        self,
        results_dir: Path,
        width: int = WINDOW_WIDTH,
        height: int = WINDOW_HEIGHT,
        fps: int = FPS,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps

        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.running = False
        self.paused = False
        self.step_once = False

        self.font_small: Optional[pygame.font.Font] = None
        self.font_subtitle: Optional[pygame.font.Font] = None
        self.font_body: Optional[pygame.font.Font] = None
        self.font_header: Optional[pygame.font.Font] = None
        self.font_title: Optional[pygame.font.Font] = None
        self.font_metric: Optional[pygame.font.Font] = None

        self.state: Dict[str, object] = {}
        self.png_path = self.results_dir / "agents_simulation.png"

        self.sim_rect = pygame.Rect(0, 0, 0, 0)
        self.sim_header_rect = pygame.Rect(0, 0, 0, 0)
        self.graph_rect = pygame.Rect(0, 0, 0, 0)
        self.sim_footer_rect = pygame.Rect(0, 0, 0, 0)
        self.side_rect = pygame.Rect(0, 0, 0, 0)
        self.chart_rect = pygame.Rect(0, 0, 0, 0)
        self.play_button = pygame.Rect(0, 0, 0, 0)
        self.pause_button = pygame.Rect(0, 0, 0, 0)
        self.step_button = pygame.Rect(0, 0, 0, 0)

        self.node_positions: Dict[str, Tuple[int, int]] = {}
        self._configure_layout()

        self.node_colors = {
            "Data": ((97, 129, 235), (38, 65, 150)),
            "Coordinator": ((233, 190, 101), (146, 93, 34)),
            "Soil": ((157, 135, 110), (94, 72, 48)),
            "Water": ((106, 168, 220), (32, 86, 138)),
            "Climate": ((124, 186, 133), (45, 99, 56)),
            "Crop": ((204, 153, 93), (119, 79, 27)),
            "Fertilizer": ((177, 141, 205), (102, 69, 131)),
            "FarmType": ((107, 183, 177), (35, 105, 100)),
            "Fuzzy": ((230, 161, 113), (138, 81, 35)),
            "Yield": ((235, 130, 96), (154, 72, 45)),
            "Optimizer": ((145, 168, 201), (59, 82, 116)),
        }
        self.phase_links = {
            "initialization": [("Data", "Coordinator"), ("Coordinator", "Crop")],
            "scenario_setup": [("Coordinator", "Crop")],
            "soil_assessment": [("Data", "Soil"), ("Soil", "Fuzzy")],
            "water_assessment": [("Data", "Water"), ("Water", "Fuzzy")],
            "climate_assessment": [("Data", "Climate"), ("Climate", "Fuzzy")],
            "farm_assessment": [("Coordinator", "FarmType"), ("FarmType", "Fuzzy")],
            "fertilizer_assessment": [("Coordinator", "Fertilizer"), ("Fertilizer", "Fuzzy")],
            "crop_assessment": [("Crop", "Fuzzy")],
            "fuzzy_assessment": [
                ("Soil", "Fuzzy"),
                ("Water", "Fuzzy"),
                ("Climate", "Fuzzy"),
                ("Crop", "Fuzzy"),
                ("Fertilizer", "Fuzzy"),
                ("FarmType", "Fuzzy"),
            ],
            "yield_assessment": [
                ("Fuzzy", "Yield"),
                ("Crop", "Yield"),
            ],
            "optimization": [("Yield", "Optimizer"), ("Optimizer", "Coordinator")],
            "recommendation": [("Optimizer", "Coordinator")],
        }

    def _configure_layout(self) -> None:
        content_top = MARGIN + HEADER_HEIGHT + 6
        top_height = self.height - content_top - MARGIN - PANEL_GAP - BOTTOM_PANEL_HEIGHT
        sim_width = self.width - (MARGIN * 2) - PANEL_GAP - PANEL_WIDTH

        self.sim_rect = pygame.Rect(MARGIN, content_top, sim_width, top_height)
        self.side_rect = pygame.Rect(self.sim_rect.right + PANEL_GAP, content_top, PANEL_WIDTH, top_height)
        self.chart_rect = pygame.Rect(MARGIN, self.sim_rect.bottom + PANEL_GAP, self.width - (MARGIN * 2), BOTTOM_PANEL_HEIGHT)

        self.sim_header_rect = pygame.Rect(
            self.sim_rect.left + SIM_PANEL_PADDING,
            self.sim_rect.top + SIM_PANEL_PADDING,
            self.sim_rect.width - (SIM_PANEL_PADDING * 2),
            SIM_HEADER_INFO_HEIGHT,
        )
        self.sim_footer_rect = pygame.Rect(
            self.sim_rect.left + SIM_PANEL_PADDING,
            self.sim_rect.bottom - SIM_PANEL_PADDING - SIM_FOOTER_INFO_HEIGHT,
            self.sim_rect.width - (SIM_PANEL_PADDING * 2),
            SIM_FOOTER_INFO_HEIGHT,
        )
        graph_top = self.sim_header_rect.bottom + SIM_SECTION_GAP
        graph_bottom = self.sim_footer_rect.top - SIM_SECTION_GAP
        self.graph_rect = pygame.Rect(
            self.sim_rect.left + GRAPH_SIDE_PADDING,
            graph_top + GRAPH_TOP_PADDING,
            self.sim_rect.width - (GRAPH_SIDE_PADDING * 2),
            max(graph_bottom - graph_top - GRAPH_TOP_PADDING, GRAPH_PANEL_MIN_HEIGHT),
        )

        button_width = BUTTON_WIDTH
        button_gap = BUTTON_GAP
        total_width = button_width * 3 + button_gap * 2
        button_left = self.side_rect.left + (self.side_rect.width - total_width) // 2
        button_top = self.side_rect.bottom - BUTTON_HEIGHT - PANEL_PADDING

        self.play_button = pygame.Rect(button_left, button_top, button_width, BUTTON_HEIGHT)
        self.pause_button = pygame.Rect(button_left + button_width + button_gap, button_top, button_width, BUTTON_HEIGHT)
        self.step_button = pygame.Rect(button_left + (button_width + button_gap) * 2, button_top, button_width, BUTTON_HEIGHT)

        left = self.graph_rect.left
        top = self.graph_rect.top
        width = self.graph_rect.width
        height = self.graph_rect.height
        self.node_positions = {
            "Data": (left + int(width * 0.12), top + int(height * 0.18)),
            "Coordinator": (left + int(width * 0.38), top + int(height * 0.16)),
            "Soil": (left + int(width * 0.14), top + int(height * 0.35)),
            "Water": (left + int(width * 0.14), top + int(height * 0.53)),
            "Climate": (left + int(width * 0.14), top + int(height * 0.72)),
            "Crop": (left + int(width * 0.40), top + int(height * 0.31)),
            "Fertilizer": (left + int(width * 0.40), top + int(height * 0.53)),
            "FarmType": (left + int(width * 0.40), top + int(height * 0.73)),
            "Fuzzy": (left + int(width * 0.67), top + int(height * 0.47)),
            "Yield": (left + int(width * 0.82), top + int(height * 0.30)),
            "Optimizer": (left + int(width * 0.82), top + int(height * 0.63)),
        }

    @staticmethod
    def _blend(a: Tuple[int, int, int], b: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
        return tuple(int((1.0 - alpha) * x + alpha * y) for x, y in zip(a, b))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def start(self, context: Dict[str, object]) -> bool:
        os.environ["SDL_VIDEO_CENTERED"] = "1"
        pygame.init()
        pygame.display.set_caption("Moroccan Agricultural Multi-Agent Simulation")
        self.window = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("segoeui", FONT_SIZE_LABEL)
        self.font_subtitle = pygame.font.SysFont("segoeui", FONT_SIZE_SUBTITLE)
        self.font_body = pygame.font.SysFont("segoeui", FONT_SIZE_MEDIUM)
        self.font_header = pygame.font.SysFont("segoeui", FONT_SIZE_LARGE, bold=True)
        self.font_title = pygame.font.SysFont("segoeui", FONT_SIZE_TITLE, bold=True)
        self.font_metric = pygame.font.SysFont("consolas", FONT_SIZE_VALUE, bold=True)

        self.running = True
        self.paused = False
        self.step_once = False
        self.state = {
            "phase": "initialization",
            "context": context,
            "scenario_name": "Loading official Moroccan data",
            "allocation_text": "Preparing candidate crop scenarios.",
            "narrative": "DataAgent is loading local Moroccan agricultural datasets.",
            "active_agents": ["Data", "Coordinator"],
            "scenario_results": [],
            "best_result": None,
            "metrics": {
                "expected_yield_t_ha": 0.0,
                "total_output_tonnes": 0.0,
                "farm_type": "pending",
                "decision_surface_ha": float(context.get("decision_surface_ha", 0.0)),
                "scenario_index": 0,
                "scenario_total": 0,
                "best_crop": "pending",
                "best_allocation": "pending",
                "fuzzy_suitability_score": 0.0,
                "resource_balance_score": max(
                    0.0,
                    min(1.0, (float(context.get("resource_balance_index", 0.4)) - 0.40) / 0.60),
                ),
                "optimization_score": 0.0,
                "agent_score": 0.0,
                "ml_predicted_yield_t_ha": 0.0,
                "ml_prediction_score": 0.0,
                "ml_model_name": "pending",
                "ml_test_mae": 0.0,
                "ml_test_rmse": 0.0,
                "ml_test_r2": 0.0,
                "ml_rmsle_display": "pending",
                "dcg_score": 0.0,
                "best_scenario_score": 0.0,
                "score_margin": 0.0,
                "water_stress": 0.0,
                "dominant_fuzzy_rule": "pending",
                "why_selected": "Optimization summary will appear here.",
            },
        }
        return self.running

    def is_running(self) -> bool:
        return self.running

    def _get_phase_timing(self, phase: str) -> Tuple[float, float]:
        if phase == "initialization":
            return DATA_LOAD_DELAY, INTERACTION_PAUSE_DELAY
        if phase == "fuzzy_assessment":
            return FUZZY_DELAY, INTERACTION_PAUSE_DELAY
        if phase == "yield_assessment":
            return YIELD_UPDATE_DELAY, INTERACTION_PAUSE_DELAY
        if phase in {"optimization", "recommendation"}:
            return OPTIMIZATION_DELAY, SCENARIO_END_DELAY
        return AGENT_ACTION_DELAY, INTERACTION_PAUSE_DELAY

    def _handle_events(self) -> None:
        if not self.running:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    self.paused = True
                    self.step_once = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.play_button.collidepoint(event.pos):
                    self.paused = False
                    self.step_once = False
                elif self.pause_button.collidepoint(event.pos):
                    self.paused = True
                elif self.step_button.collidepoint(event.pos):
                    self.paused = True
                    self.step_once = True

    def _wrap_lines(
        self,
        text: str,
        font: pygame.font.Font,
        width: int,
        max_lines: Optional[int] = 3,
    ) -> List[str]:
        words = text.split()
        if not words:
            return [""]

        lines: List[str] = []
        current = ""
        for index, word in enumerate(words):
            candidate = f"{current} {word}".strip()
            if not current or font.size(candidate)[0] <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
            if max_lines is not None and len(lines) >= max_lines - 1:
                remainder_words = [current] + words[index + 1 :]
                tail = " ".join(remainder_words).strip()
                while font.size(f"{tail}...")[0] > width and " " in tail:
                    tail = " ".join(tail.split()[:-1]).strip()
                if tail != " ".join(remainder_words).strip() or index + 1 < len(words):
                    tail = f"{tail}..."
                lines.append(tail)
                return lines[:max_lines]

        if current:
            lines.append(current)
        return lines if max_lines is None else lines[:max_lines]

    def _measure_wrapped_text_height(
        self,
        text: str,
        font: pygame.font.Font,
        width: int,
        max_lines: Optional[int] = 3,
        min_lines: int = 1,
    ) -> int:
        lines = self._wrap_lines(str(text), font, width, max_lines)
        line_count = max(len(lines), min_lines)
        return line_count * (font.get_linesize() + 2)

    def _draw_text(
        self,
        text: str,
        pos: Tuple[int, int],
        font: pygame.font.Font,
        color: Tuple[int, int, int],
        center: bool = False,
    ) -> None:
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=pos) if center else surface.get_rect(topleft=pos)
        self.window.blit(surface, rect)

    def _draw_wrapped_text(
        self,
        text: str,
        rect: pygame.Rect,
        font: pygame.font.Font,
        color: Tuple[int, int, int],
        max_lines: int = 3,
    ) -> None:
        y = rect.top
        for line in self._wrap_lines(str(text), font, rect.width, max_lines):
            rendered = font.render(line, True, color)
            self.window.blit(rendered, (rect.left, y))
            y += rendered.get_height() + 2

    def _draw_panel(self, rect: pygame.Rect, fill: Tuple[int, int, int], border: Tuple[int, int, int]) -> None:
        pygame.draw.rect(self.window, fill, rect, border_radius=18)
        pygame.draw.rect(self.window, border, rect, width=2, border_radius=18)

    def _draw_button(self, rect: pygame.Rect, label: str, active: bool, accent: Tuple[int, int, int]) -> None:
        fill = self._blend((241, 245, 249), accent, 0.32 if active else 0.10)
        border = accent if active else (151, 162, 178)
        pygame.draw.rect(self.window, fill, rect, border_radius=10)
        pygame.draw.rect(self.window, border, rect, width=1, border_radius=10)
        self._draw_text(label, rect.center, self.font_small, (28, 40, 60), center=True)

    def _phase_display(self, phase: str) -> str:
        return phase.replace("_", " ").title()

    def _arrow_head(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction) or 1.0
        unit = direction / length
        tip = np.array(end)
        base = tip - unit * 12
        normal = np.array([-unit[1], unit[0]]) * 6
        points = [tuple(tip.astype(int)), tuple((base + normal).astype(int)), tuple((base - normal).astype(int))]
        pygame.draw.polygon(self.window, color, points)

    def _draw_connection(self, start: str, end: str, progress: float, highlighted: bool) -> None:
        start_pos = self.node_positions[start]
        end_pos = self.node_positions[end]
        base_color = (191, 198, 209)
        active_color = (235, 169, 66)
        pygame.draw.line(self.window, base_color, start_pos, end_pos, 2)
        self._arrow_head(start_pos, end_pos, base_color)

        if highlighted:
            pulse = (
                int(start_pos[0] + (end_pos[0] - start_pos[0]) * progress),
                int(start_pos[1] + (end_pos[1] - start_pos[1]) * progress),
            )
            pygame.draw.line(self.window, active_color, start_pos, pulse, 4)
            pygame.draw.circle(self.window, active_color, pulse, 8)
            self._arrow_head(start_pos, end_pos, active_color)

    def _draw_active_halo(self, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], progress: float) -> None:
        halo_radius = radius + int(5 + 5 * np.sin(progress * np.pi))
        halo_surface = pygame.Surface((halo_radius * 2 + 8, halo_radius * 2 + 8), pygame.SRCALPHA)
        pygame.draw.circle(halo_surface, (*color, 56), (halo_radius + 4, halo_radius + 4), halo_radius)
        pygame.draw.circle(halo_surface, (*color, 120), (halo_radius + 4, halo_radius + 4), halo_radius - 3, width=3)
        self.window.blit(halo_surface, (center[0] - halo_radius - 4, center[1] - halo_radius - 4))

    def _draw_node(self, name: str, label: str, active: bool, progress: float) -> None:
        x, y = self.node_positions[name]
        fill, border = self.node_colors[name]
        rect = pygame.Rect(0, 0, AGENT_NODE_WIDTH, AGENT_NODE_HEIGHT)
        rect.center = (x, y)
        fill_color = self._blend(fill, (255, 225, 150), 0.28) if active else fill
        if active:
            self._draw_active_halo((x, y), 32, (255, 211, 101), progress)
        pygame.draw.rect(self.window, fill_color, rect, border_radius=14)
        pygame.draw.rect(self.window, border, rect, width=2, border_radius=14)
        label_font = self.font_body if self.font_body.size(label)[0] <= rect.width - 16 else self.font_small
        self._draw_text(label, (x, y - 6), label_font, (24, 35, 52), center=True)
        if active:
            self._draw_text("ACTIVE", (x, y + 10), self.font_small, (60, 72, 91), center=True)

    def _draw_simulation_area(self, progress: float) -> None:
        rect = self.sim_rect
        self._draw_panel(rect, (248, 250, 252), (204, 214, 226))
        phase = str(self.state.get("phase", "initialization"))
        active_agents = list(self.state.get("active_agents", []))
        highlighted = set(self.phase_links.get(phase, []))

        header_rect = self.sim_header_rect
        footer_rect = self.sim_footer_rect
        graph_rect = self.graph_rect

        pygame.draw.rect(self.window, (236, 241, 247), header_rect, border_radius=14)
        pygame.draw.rect(self.window, (194, 205, 219), header_rect, width=2, border_radius=14)
        pygame.draw.rect(self.window, (255, 255, 255), graph_rect, border_radius=16)
        pygame.draw.rect(self.window, (212, 222, 233), graph_rect, width=2, border_radius=16)
        pygame.draw.rect(self.window, (236, 241, 247), footer_rect, border_radius=14)
        pygame.draw.rect(self.window, (194, 205, 219), footer_rect, width=2, border_radius=14)

        self._draw_text("Agent Graph", (header_rect.left + 12, header_rect.top + 7), self.font_header, (24, 38, 59))

        active_box = pygame.Rect(
            header_rect.left + 12,
            header_rect.top + 28,
            int(header_rect.width * 0.60),
            STATUS_BOX_HEIGHT,
        )
        phase_box = pygame.Rect(
            active_box.right + 10,
            active_box.top,
            header_rect.right - 12 - (active_box.right + 10),
            STATUS_BOX_HEIGHT,
        )
        flow_box = pygame.Rect(
            header_rect.left + 12,
            header_rect.bottom - FLOW_HINT_HEIGHT - 6,
            header_rect.width - 24,
            FLOW_HINT_HEIGHT,
        )

        for info_rect in (active_box, phase_box, flow_box):
            pygame.draw.rect(self.window, (247, 250, 253), info_rect, border_radius=10)
            pygame.draw.rect(self.window, (196, 206, 219), info_rect, width=1, border_radius=10)

        active_text = ", ".join(active_agents) if active_agents else "Waiting"
        self._draw_text(
            f"Active: {active_text}",
            (active_box.left + 10, active_box.top + 6),
            self.font_small,
            (44, 57, 76),
        )
        phase_badge = f"{self._phase_display(phase)} | {len(highlighted)} active links"
        self._draw_text(
            phase_badge,
            (phase_box.left + 10, phase_box.top + 6),
            self.font_small,
            (52, 66, 86),
        )
        self._draw_text("Protected graph area for agent labels and links", (flow_box.left + 8, flow_box.top + 2), self.font_small, (67, 79, 97))

        default_links = {
            ("Data", "Coordinator"),
            ("Data", "Soil"),
            ("Data", "Water"),
            ("Data", "Climate"),
            ("Coordinator", "Crop"),
            ("Coordinator", "FarmType"),
            ("Coordinator", "Fertilizer"),
            ("Soil", "Fuzzy"),
            ("Water", "Fuzzy"),
            ("Climate", "Fuzzy"),
            ("Crop", "Fuzzy"),
            ("Fertilizer", "Fuzzy"),
            ("FarmType", "Fuzzy"),
            ("Fuzzy", "Yield"),
            ("Yield", "Optimizer"),
            ("Optimizer", "Coordinator"),
        }
        for start, end in default_links:
            self._draw_connection(start, end, progress, (start, end) in highlighted)

        labels = {
            "Data": "Data",
            "Coordinator": "Coordinator",
            "Soil": "Soil",
            "Water": "Water",
            "Climate": "Climate",
            "Crop": "Crop",
            "Fertilizer": "Fertilizer",
            "FarmType": "Farm Type",
            "Fuzzy": "Fuzzy Logic",
            "Yield": "Yield",
            "Optimizer": "Optimizer",
        }
        for name, label in labels.items():
            self._draw_node(name, label, name in active_agents, progress)

        scenario_rect = pygame.Rect(footer_rect.left + 10, footer_rect.top + 7, footer_rect.width - 20, footer_rect.height - 14)
        pygame.draw.rect(self.window, (247, 250, 253), scenario_rect, border_radius=12)
        pygame.draw.rect(self.window, (191, 202, 217), scenario_rect, width=1, border_radius=12)
        self._draw_text("Current Scenario", (scenario_rect.left + 8, scenario_rect.top + 5), self.font_small, (46, 58, 76))
        self._draw_wrapped_text(
            str(self.state.get("allocation_text", "")),
            pygame.Rect(scenario_rect.left + 8, scenario_rect.top + 20, scenario_rect.width - 16, scenario_rect.height - 24),
            self.font_small,
            (32, 44, 62),
            max_lines=2,
        )

    def _draw_metric_tile(
        self,
        rect: pygame.Rect,
        label: str,
        value: str,
        accent: Tuple[int, int, int],
        emphasize: bool = False,
    ) -> None:
        pygame.draw.rect(self.window, self._blend((255, 255, 255), accent, 0.10), rect, border_radius=12)
        pygame.draw.rect(self.window, accent, rect, width=1, border_radius=12)
        label_lines = self._wrap_lines(label, self.font_small, rect.width - 14, max_lines=2)
        label_y = rect.top + 5
        for line in label_lines:
            self._draw_text(line, (rect.left + 7, label_y), self.font_small, (54, 65, 83))
            label_y += self.font_small.get_linesize() + 1
        font = self.font_metric if emphasize else self.font_body
        if font.size(value)[0] > rect.width - 12:
            font = self.font_body if emphasize else self.font_small
        available_top = label_y + 2
        available_height = max(rect.bottom - available_top - 4, self.font_small.get_linesize())
        if font.get_linesize() > available_height:
            font = self.font_body if font is self.font_metric else self.font_small
        value_center_y = min(available_top + available_height // 2, rect.bottom - 12)
        self._draw_text(value, (rect.centerx, value_center_y), font, (22, 35, 52), center=True)

    def _draw_factor_chip(self, rect: pygame.Rect, label: str, value: float, accent: Tuple[int, int, int]) -> None:
        pygame.draw.rect(self.window, self._blend((255, 255, 255), accent, 0.12), rect, border_radius=10)
        pygame.draw.rect(self.window, accent, rect, width=1, border_radius=10)
        self._draw_text(label, (rect.left + 6, rect.top + 4), self.font_small, (50, 62, 80))
        value_text = f"{value:.2f}"
        value_width = self.font_small.size(value_text)[0]
        self._draw_text(value_text, (rect.right - value_width - 6, rect.top + 4), self.font_small, (24, 36, 54))

    def _draw_side_panel(self) -> None:
        rect = self.side_rect
        panel_pad = SIDE_PANEL_PADDING
        card_gap = 6
        self._draw_panel(rect, (247, 249, 252), (204, 214, 226))
        context = self.state.get("context", {}) or {}
        metrics = self.state.get("metrics", {}) or {}
        best_result = self.state.get("best_result") or {}

        content_left = rect.left + panel_pad
        content_width = rect.width - (panel_pad * 2)
        button_top = self.play_button.top
        y = rect.top + 14

        self._draw_text("Decision Panel", (content_left + 2, y), self.font_header, (24, 38, 59))
        y += self.font_header.get_linesize() + 2
        self._draw_wrapped_text(
            f"{context.get('region', 'National')} Morocco | {context.get('year', '--')}",
            pygame.Rect(content_left + 2, y, content_width - 4, self.font_small.get_linesize() + 4),
            self.font_small,
            (79, 92, 111),
            max_lines=1,
        )
        y += self.font_small.get_linesize() + card_gap + 2

        scenario_card = pygame.Rect(content_left, y, content_width, 62)
        pygame.draw.rect(self.window, (232, 238, 245), scenario_card, border_radius=14)
        pygame.draw.rect(self.window, (183, 195, 212), scenario_card, width=1, border_radius=14)
        scenario_index = int(metrics.get("scenario_index", 0))
        scenario_total = int(metrics.get("scenario_total", 0))
        scenario_meta = (
            f"Scenario {scenario_index}/{scenario_total} | {self._phase_display(str(self.state.get('phase', 'initialization')))}"
            if scenario_total
            else self._phase_display(str(self.state.get("phase", "initialization")))
        )
        self._draw_text("Current Scenario", (scenario_card.left + 10, scenario_card.top + 8), self.font_small, (46, 58, 76))
        self._draw_wrapped_text(
            scenario_meta,
            pygame.Rect(scenario_card.left + 10, scenario_card.top + 22, scenario_card.width - 20, 14),
            self.font_small,
            (70, 83, 101),
            max_lines=1,
        )
        self._draw_wrapped_text(
            str(self.state.get("scenario_name", "--")),
            pygame.Rect(scenario_card.left + 10, scenario_card.top + 36, scenario_card.width - 20, 14),
            self.font_small,
            (28, 40, 60),
            max_lines=1,
        )
        y = scenario_card.bottom + card_gap

        recommendation_card = pygame.Rect(content_left, y, content_width, 56)
        pygame.draw.rect(self.window, (229, 236, 245), recommendation_card, border_radius=14)
        pygame.draw.rect(self.window, (173, 188, 210), recommendation_card, width=1, border_radius=14)
        self._draw_text("Best Current Recommendation", (recommendation_card.left + 10, recommendation_card.top + 8), self.font_small, (46, 58, 76))
        self._draw_wrapped_text(
            f"Best crop: {metrics.get('best_crop', best_result.get('focus_crop', 'pending'))} | "
            f"Farm: {str(metrics.get('farm_type', best_result.get('farm_type', '--'))).title()}",
            pygame.Rect(recommendation_card.left + 10, recommendation_card.top + 22, recommendation_card.width - 20, 14),
            self.font_small,
            (54, 66, 84),
            max_lines=1,
        )
        self._draw_wrapped_text(
            str(metrics.get("best_allocation", best_result.get("allocation_text", "pending"))),
            pygame.Rect(recommendation_card.left + 10, recommendation_card.top + 34, recommendation_card.width - 20, 14),
            self.font_small,
            (28, 40, 60),
            max_lines=1,
        )
        y = recommendation_card.bottom + card_gap

        tile_gap = card_gap
        tile_width = (content_width - (tile_gap * 2)) // 3
        tile_height = 42
        metric_tiles = [
            ("Expected Yield", f"{self._safe_float(metrics.get('expected_yield_t_ha')):.2f} t/ha", (220, 74, 58), True),
            ("Expected Output", f"{self._safe_float(metrics.get('total_output_tonnes')):.1f} t", (29, 78, 216), True),
            ("ML Predicted", f"{self._safe_float(metrics.get('ml_predicted_yield_t_ha')):.2f} t/ha", (98, 87, 201), False),
            ("Fuzzy Score", f"{self._safe_float(metrics.get('fuzzy_suitability_score')):.2f}", (212, 125, 54), False),
            ("Resource Balance", f"{self._safe_float(metrics.get('resource_balance_score')):.2f}", (16, 163, 127), False),
            ("Hybrid Score", f"{self._safe_float(metrics.get('optimization_score')):.2f}", (88, 102, 190), False),
        ]
        for index, (label, value, accent, emphasize) in enumerate(metric_tiles):
            row = index // 3
            col = index % 3
            tile_rect = pygame.Rect(
                content_left + col * (tile_width + tile_gap),
                y + row * (tile_height + tile_gap),
                tile_width,
                tile_height,
            )
            self._draw_metric_tile(tile_rect, label, value, accent, emphasize)
        y += (tile_height * 2) + tile_gap

        factor_card = pygame.Rect(content_left, y, content_width, 38)
        pygame.draw.rect(self.window, (233, 239, 246), factor_card, border_radius=12)
        pygame.draw.rect(self.window, (183, 195, 212), factor_card, width=1, border_radius=12)
        self._draw_text("Contribution Scores", (factor_card.left + 10, factor_card.top + 5), self.font_small, (46, 58, 76))
        factor_values = [
            ("Soil", self._safe_float(metrics.get("average_soil_score")), (140, 111, 81)),
            ("Water", self._safe_float(metrics.get("average_water_score")), (67, 124, 189)),
            ("Climate", self._safe_float(metrics.get("average_climate_score")), (74, 139, 84)),
            ("Fertilizer", self._safe_float(metrics.get("average_fertilizer_score")), (139, 99, 171)),
            ("Compat", self._safe_float(metrics.get("average_compatibility_score")), (229, 123, 80)),
        ]
        chip_gap = 4
        chip_width = (factor_card.width - 24 - chip_gap * 4) // 5
        for index, (label, value, accent) in enumerate(factor_values):
            chip_rect = pygame.Rect(
                factor_card.left + 10 + index * (chip_width + chip_gap),
                factor_card.top + 18,
                chip_width,
                16,
            )
            self._draw_factor_chip(chip_rect, label, value, accent)
        y = factor_card.bottom + card_gap

        why_card = pygame.Rect(content_left, y, content_width, 48)
        pygame.draw.rect(self.window, (231, 237, 245), why_card, border_radius=14)
        pygame.draw.rect(self.window, (181, 194, 212), why_card, width=1, border_radius=14)
        self._draw_text("Why This Scenario Won", (why_card.left + 10, why_card.top + 6), self.font_small, (46, 58, 76))
        self._draw_wrapped_text(
            str(metrics.get("why_selected", best_result.get("win_reason", "Ranking in progress."))),
            pygame.Rect(why_card.left + 10, why_card.top + 20, why_card.width - 20, 16),
            self.font_small,
            (40, 52, 70),
            max_lines=1,
        )
        y = why_card.bottom + card_gap

        decision_card_height = max(button_top - y - 6, 36)
        decision_card = pygame.Rect(content_left, y, content_width, decision_card_height)
        pygame.draw.rect(self.window, (233, 238, 246), decision_card, border_radius=12)
        pygame.draw.rect(self.window, (183, 195, 212), decision_card, width=1, border_radius=12)
        self._draw_text("Decision Metrics", (decision_card.left + 10, decision_card.top + 5), self.font_small, (46, 58, 76))

        dcg_text = f"DCG Score: {self._safe_float(metrics.get('dcg_score')):.2f}"
        best_score_text = f"Best Scenario Score: {self._safe_float(metrics.get('best_scenario_score')):.2f}"
        margin_text = f"Score Margin: {self._safe_float(metrics.get('score_margin')):.2f}"
        if decision_card.height >= 60:
            self._draw_text(dcg_text, (decision_card.left + 10, decision_card.top + 20), self.font_small, (40, 52, 70))
            self._draw_text(best_score_text, (decision_card.left + 10, decision_card.top + 34), self.font_small, (40, 52, 70))
            self._draw_text(margin_text, (decision_card.left + 10, decision_card.top + 48), self.font_small, (40, 52, 70))
        else:
            compact_metrics = f"{dcg_text} | {best_score_text} | {margin_text}"
            self._draw_wrapped_text(
                compact_metrics,
                pygame.Rect(decision_card.left + 10, decision_card.top + 20, decision_card.width - 20, decision_card.height - 24),
                self.font_small,
                (40, 52, 70),
                max_lines=2,
            )

        self._draw_button(self.play_button, "Play", not self.paused, (34, 197, 94))
        self._draw_button(self.pause_button, "Pause", self.paused and not self.step_once, (239, 68, 68))
        self._draw_button(self.step_button, "Step", self.step_once, (59, 130, 246))

    def _draw_chart(self) -> None:
        rect = self.chart_rect
        self._draw_panel(rect, (248, 250, 252), (204, 214, 226))
        self._draw_text("Scenario Comparison", (rect.left + CHART_PANEL_PADDING, rect.top + 10), self.font_header, (24, 38, 59))
        self._draw_text(
            "Output and hybrid score after multi-agent reasoning plus supervised regression",
            (rect.left + CHART_PANEL_PADDING, rect.top + 30),
            self.font_small,
            (82, 95, 114),
        )

        plot_rect = pygame.Rect(
            rect.left + CHART_PANEL_PADDING,
            rect.top + 48,
            rect.width - (CHART_PANEL_PADDING * 2),
            rect.height - 58,
        )
        pygame.draw.rect(self.window, (255, 255, 255), plot_rect, border_radius=14)
        pygame.draw.rect(self.window, (214, 221, 230), plot_rect, width=1, border_radius=14)

        scenario_results = list(self.state.get("scenario_results", []))[:3]
        if not scenario_results:
            self._draw_text(
                "Scenario outputs and optimization ranking will appear here during the live simulation.",
                (plot_rect.left + 14, plot_rect.top + 16),
                self.font_small,
                (95, 105, 124),
            )
            return

        header_y = plot_rect.top + 8
        self._draw_text("Rank", (plot_rect.left + 12, header_y), self.font_small, (65, 78, 97))
        self._draw_text("Scenario", (plot_rect.left + 58, header_y), self.font_small, (65, 78, 97))
        self._draw_text("Output", (plot_rect.left + plot_rect.width - 178, header_y), self.font_small, (65, 78, 97))
        self._draw_text("Score", (plot_rect.left + plot_rect.width - 84, header_y), self.font_small, (65, 78, 97))

        max_output = max(float(item["total_expected_output_tonnes"]) for item in scenario_results)
        row_height = 20
        for index, item in enumerate(scenario_results):
            row_top = plot_rect.top + 24 + index * row_height
            row_rect = pygame.Rect(plot_rect.left + 6, row_top, plot_rect.width - 12, row_height - 2)
            is_best = int(item.get("rank", index + 1)) == 1
            fill = (238, 243, 249) if not is_best else (248, 233, 224)
            border = (217, 224, 233) if not is_best else (221, 164, 133)
            pygame.draw.rect(self.window, fill, row_rect, border_radius=10)
            pygame.draw.rect(self.window, border, row_rect, width=1, border_radius=10)

            self._draw_text(f"#{int(item.get('rank', index + 1))}", (row_rect.left + 6, row_rect.top + 3), self.font_small, (42, 54, 72))
            self._draw_wrapped_text(
                str(item["scenario_name"]),
                pygame.Rect(row_rect.left + 40, row_rect.top + 3, row_rect.width - 206, 16),
                self.font_small,
                (42, 54, 72),
                max_lines=1,
            )
            self._draw_text(
                f"{float(item['total_expected_output_tonnes']):.0f} t",
                (row_rect.left + row_rect.width - 156, row_rect.top + 3),
                self.font_small,
                (42, 54, 72),
            )
            self._draw_text(
                f"{float(item.get('optimization_score', 0.0)):.2f}",
                (row_rect.left + row_rect.width - 62, row_rect.top + 3),
                self.font_small,
                (42, 54, 72),
            )

            bar_left = row_rect.left + row_rect.width - 144
            bar_top = row_rect.bottom - 6
            bar_width = int(108 * (float(item["total_expected_output_tonnes"]) / max(max_output, 1.0)))
            pygame.draw.line(self.window, (224, 229, 237), (bar_left, bar_top), (bar_left + 108, bar_top), 4)
            pygame.draw.line(
                self.window,
                (76, 124, 218) if not is_best else (233, 126, 93),
                (bar_left, bar_top),
                (bar_left + bar_width, bar_top),
                4,
            )

    def _render(self, progress: float) -> None:
        top_color = np.array([239, 244, 249], dtype=float)
        bottom_color = np.array([227, 233, 241], dtype=float)
        for y in range(self.height):
            alpha = y / max(self.height - 1, 1)
            color = self._blend(tuple(top_color.astype(int)), tuple(bottom_color.astype(int)), alpha)
            pygame.draw.line(self.window, color, (0, y), (self.width, y))

        self._draw_text("Moroccan Agricultural Optimization Simulation", (MARGIN, 12), self.font_title, (19, 32, 54))
        self._draw_text(
            "Fixed 1280x720 live demo with normalized scoring, fuzzy logic, supervised regression, and allocation optimization",
            (MARGIN + 2, 38),
            self.font_subtitle,
            (80, 95, 116),
        )

        metrics = self.state.get("metrics", {}) or {}
        phase_text = self._phase_display(str(self.state.get("phase", "initialization")))
        scenario_index = int(metrics.get("scenario_index", 0))
        scenario_total = int(metrics.get("scenario_total", 0))
        badge_text = f"{phase_text} | Scenario {scenario_index}/{scenario_total}" if scenario_total else phase_text
        badge_rect = pygame.Rect(self.width - 258, 14, 246, 30)
        pygame.draw.rect(self.window, (231, 239, 246), badge_rect, border_radius=12)
        pygame.draw.rect(self.window, (179, 193, 211), badge_rect, width=1, border_radius=12)
        self._draw_text(badge_text, badge_rect.center, self.font_small, (34, 49, 68), center=True)

        self._draw_simulation_area(progress)
        self._draw_side_panel()
        self._draw_chart()
        pygame.display.flip()

    def _hold_current_frame(self, duration: float) -> None:
        elapsed = 0.0
        while self.running and elapsed < duration:
            dt = self.clock.tick(self.fps) / 1000.0
            self._handle_events()
            if not self.running:
                break
            if self.paused and not self.step_once:
                self._render(1.0)
                continue
            elapsed += dt
            self._render(1.0)
            if self.step_once:
                self.step_once = False
                break

    def update(
        self,
        phase: str,
        scenario_name: str,
        allocation_text: str,
        narrative: str,
        metrics: Dict[str, object],
        scenario_results: List[Dict[str, object]],
        best_result: Optional[Dict[str, object]],
        active_agents: Optional[Iterable[str]] = None,
    ) -> bool:
        if not self.running or self.window is None:
            return False

        self.state = {
            **self.state,
            "phase": phase,
            "scenario_name": scenario_name,
            "allocation_text": allocation_text,
            "narrative": narrative,
            "metrics": dict(metrics),
            "scenario_results": list(scenario_results),
            "best_result": dict(best_result) if best_result else None,
            "active_agents": list(active_agents or []),
        }

        action_duration, hold_duration = self._get_phase_timing(phase)
        progress = 0.0

        while self.running and progress < 1.0:
            dt = self.clock.tick(self.fps) / 1000.0
            self._handle_events()
            if not self.running:
                break
            if self.paused and not self.step_once:
                self._render(progress)
                continue
            progress = min(1.0, progress + dt / max(action_duration, 0.001))
            self._render(progress)
            if self.step_once and progress >= 1.0:
                self.step_once = False

        if self.running and hold_duration > 0.0:
            self._hold_current_frame(hold_duration)

        return self.running

    def finalize(self) -> Dict[str, Path]:
        outputs = {"png": self.png_path}
        if self.window is not None:
            self._render(1.0)
            pygame.image.save(self.window, self.png_path)
            pygame.quit()
            self.running = False
        return outputs
