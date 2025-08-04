"""
Enhanced panel classes for the STAG Visualizer with beautiful styling and improved visibility.
"""

import pygame
import numpy as np
import networkx as nx
import math
from typing import Dict, List, Tuple, Optional, Any
from .config import *
from .ui_components import *

class GridPanel:
    """Enhanced grid panel with smooth animations and beautiful cell rendering."""

    def __init__(self, size: int):
        self.size = size
        self.width = self.height = size * CELL_SIZE
        self.surface = pygame.Surface((self.width, self.height))
        self.cell_animations = {}  # Track cell animation states

    def update(self, frame: int):
        """Update animations."""
        # Update cell animations
        for pos in list(self.cell_animations.keys()):
            self.cell_animations[pos] = max(0, self.cell_animations[pos] - 1)
            if self.cell_animations[pos] == 0:
                del self.cell_animations[pos]

    def draw(self, state: Dict[str, Any]) -> pygame.Surface:
        """Draw the enhanced grid with beautiful styling."""
        # Clear with gradient background
        draw_gradient_rect(self.surface, COLORS['panel_bg'],
                          (30, 35, 45), pygame.Rect(0, 0, self.width, self.height))

        if 'observation' not in state:
            self._draw_waiting_state()
            return self.surface

        obs = np.array(state['observation'])

        # Draw cells with enhanced styling
        for r in range(self.size):
            for c in range(self.size):
                self._draw_enhanced_cell(obs[r, c], r, c, state)

        # Draw grid lines
        self._draw_grid_lines()

        # Draw border
        pygame.draw.rect(self.surface, COLORS['panel_border'],
                        pygame.Rect(0, 0, self.width, self.height), 2)

        return self.surface

    def _draw_enhanced_cell(self, cell_value: int, row: int, col: int, state: Dict[str, Any]):
        """Draw a single cell with enhanced visual effects."""
        x, y = col * CELL_SIZE, row * CELL_SIZE
        cell_rect = pygame.Rect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2)

        # Get base color
        base_color = ENTITY_COLOR_MAP.get(cell_value, COLORS['empty'])

        # Add animation effects
        if (row, col) in self.cell_animations:
            base_color = pulse_color(base_color, self.cell_animations[(row, col)], 0.2, 0.5)

        # Draw cell with rounded corners
        draw_rounded_rect(self.surface, base_color, cell_rect, 3)

        # Add highlight for special cells
        if cell_value != 0:  # Not empty
            highlight_rect = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            highlight_color = (*COLORS['highlight'][:3], 30)
            highlight_surface = pygame.Surface((CELL_SIZE - 4, CELL_SIZE - 4), pygame.SRCALPHA)
            highlight_surface.fill(highlight_color)
            self.surface.blit(highlight_surface, (x + 2, y + 2))

        # Draw entity-specific details
        self._draw_entity_details(cell_value, x, y)

    def _draw_entity_details(self, cell_value: int, x: int, y: int):
        """Draw specific details for different entity types."""
        center_x, center_y = x + CELL_SIZE // 2, y + CELL_SIZE // 2

        if cell_value == 1:  # Player
            # Draw player as a glowing circle
            draw_glowing_circle(self.surface, COLORS['player'], (center_x, center_y), 8, 15)
        elif cell_value == 2:  # Goal
            # Draw goal as a star
            self._draw_star(center_x, center_y, 10, COLORS['goal'])
        elif cell_value == 3:  # Obstacle
            # Draw obstacle as a diamond
            self._draw_diamond(center_x, center_y, 12, COLORS['obstacle'])

    def _draw_star(self, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw a star shape."""
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            radius = size if i % 2 == 0 else size // 2
            px = x + radius * math.cos(angle - math.pi / 2)
            py = y + radius * math.sin(angle - math.pi / 2)
            points.append((px, py))
        pygame.draw.polygon(self.surface, color, points)

    def _draw_diamond(self, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw a diamond shape."""
        points = [
            (x, y - size),
            (x + size, y),
            (x, y + size),
            (x - size, y)
        ]
        pygame.draw.polygon(self.surface, color, points)

    def _draw_grid_lines(self):
        """Draw subtle grid lines."""
        line_color = COLORS['border_dark']
        for i in range(self.size + 1):
            # Vertical lines
            pygame.draw.line(self.surface, line_color,
                           (i * CELL_SIZE, 0), (i * CELL_SIZE, self.height), 1)
            # Horizontal lines
            pygame.draw.line(self.surface, line_color,
                           (0, i * CELL_SIZE), (self.width, i * CELL_SIZE), 1)

    def _draw_waiting_state(self):
        """Draw waiting state with animated text."""
        font = get_font(LARGE_FONT_SIZE)
        text = "Waiting for data..."
        text_surface = create_text_with_shadow(text, font, COLORS['text_secondary'])

        # Center the text
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.surface.blit(text_surface, text_rect)

class InfoPanel:
    """Enhanced information panel with beautiful metrics display."""

    def __init__(self):
        self.surface = pygame.Surface((GRID_PANEL_WIDTH, INFO_PANEL_HEIGHT))
        self.metrics_history = []
        self.max_history = 100

    def update(self, frame: int):
        """Update panel animations and data."""
        pass

    def draw(self, state: Dict[str, Any], graph: nx.Graph = None) -> pygame.Surface:
        """Draw the enhanced info panel."""
        # Clear with gradient background
        draw_gradient_rect(self.surface, COLORS['panel_bg'],
                          (25, 30, 40), pygame.Rect(0, 0, GRID_PANEL_WIDTH, INFO_PANEL_HEIGHT))

        if not state:
            self._draw_waiting_message()
            return self.surface

        # Draw metrics with enhanced styling
        self._draw_metrics(state, graph)

        # Draw border
        pygame.draw.rect(self.surface, COLORS['panel_border'],
                        pygame.Rect(0, 0, GRID_PANEL_WIDTH, INFO_PANEL_HEIGHT), 2)

        return self.surface

    def _draw_metrics(self, state: Dict[str, Any], graph: nx.Graph):
        """Draw metrics with enhanced visual styling."""
        font = get_font(FONT_SIZE)
        small_font = get_font(SMALL_FONT_SIZE)

        # Prepare metrics data
        metrics = [
            ("Episode", state.get('episode', 'N/A'), COLORS['primary']),
            ("Step", state.get('step', 'N/A'), COLORS['secondary']),
            ("Active Node", state.get('active_node_id', 'N/A'), COLORS['accent']),
            ("Total Nodes", graph.number_of_nodes() if graph else 0, COLORS['success']),
            ("Action", state.get('action', 'N/A'), COLORS['warning']),
            ("Reward", f"{state.get('reward', 0.0):.3f}", COLORS['danger']),
            ("Total Reward", f"{state.get('total_reward', 0.0):.2f}", COLORS['success']),
        ]

        # Draw metrics in a grid layout
        cols = 3
        col_width = GRID_PANEL_WIDTH // cols
        row_height = 25

        for i, (label, value, color) in enumerate(metrics):
            col = i % cols
            row = i // cols

            x = col * col_width + 10
            y = row * row_height + 10

            # Draw label
            label_surface = small_font.render(f"{label}:", True, COLORS['text_secondary'])
            self.surface.blit(label_surface, (x, y))

            # Draw value with color coding
            value_surface = font.render(str(value), True, color)
            self.surface.blit(value_surface, (x, y + 12))

    def _draw_waiting_message(self):
        """Draw waiting message."""
        font = get_font(FONT_SIZE)
        text = "Waiting for training data..."
        text_surface = create_text_with_shadow(text, font, COLORS['text_secondary'])

        text_rect = text_surface.get_rect(center=(GRID_PANEL_WIDTH // 2, INFO_PANEL_HEIGHT // 2))
        self.surface.blit(text_surface, text_rect)

class GraphPanel:
    """Enhanced 3D graph panel with beautiful network visualization and improved connection visibility."""

    def __init__(self):
        self.surface = pygame.Surface((GRAPH_PANEL_WIDTH, (11 * CELL_SIZE) + INFO_PANEL_HEIGHT))
        self.graph = nx.Graph()
        self.node_positions_3d = None
        self.rotation_angle = 0
        self.paused = False
        self.frame_count = 0

        # Enhanced visualization settings
        self.camera_distance = GRAPH_3D_CONFIG['camera_distance']
        self.scale = GRAPH_3D_CONFIG['scale']
        self.rotation_x = GRAPH_3D_CONFIG['rotation_x']

        # Connection visibility enhancements
        self.edge_weights = {}
        self.edge_activities = {}

    def update(self, frame: int):
        """Update animations and rotations."""
        self.frame_count = frame
        if not self.paused:
            self.rotation_angle += ROTATION_SPEED

    def toggle_pause(self):
        """Toggle rotation pause."""
        self.paused = not self.paused

    def reset_rotation(self):
        """Reset rotation angle."""
        self.rotation_angle = 0

    def update_data(self, state: Dict[str, Any]):
        """Update graph data with enhanced connection tracking."""
        graph_state = state.get('graph_state', {})
        if not graph_state or not graph_state.get('nodes'):
            return

        self.graph.clear()
        self.edge_weights.clear()
        self.edge_activities.clear()

        # Add nodes with enhanced attributes
        for node in graph_state.get('nodes', []):
            node_id = node['id']
            node_type = node.get('type', 'sensory')
            activation = node.get('activation', 0.0)

            self.graph.add_node(node_id,
                              type=node_type,
                              activation=activation,
                              size=self._calculate_node_size(activation))

        # Add edges with enhanced attributes
        for link in graph_state.get('links', []):
            source, target = link['source'], link['target']
            weight = link.get('weight', 1.0)
            activity = link.get('activity', 0.0)

            self.graph.add_edge(source, target, weight=weight, activity=activity)
            self.edge_weights[(source, target)] = weight
            self.edge_activities[(source, target)] = activity

        # Recalculate 3D positions if graph structure changed
        if self.graph.number_of_nodes() > 0:
            self.node_positions_3d = nx.spring_layout(self.graph, dim=3, seed=42, k=2.0)

    def draw(self, state: Dict[str, Any]) -> pygame.Surface:
        """Draw the enhanced 3D graph visualization."""
        # Clear with gradient background
        draw_gradient_rect(self.surface, (15, 20, 30), (25, 30, 40),
                          pygame.Rect(0, 0, GRAPH_PANEL_WIDTH, self.surface.get_height()))

        if not self.node_positions_3d or self.graph.number_of_nodes() == 0:
            self._draw_empty_state()
            return self.surface

        # Calculate 3D projections
        projected_points = self._calculate_projections()

        # Draw enhanced connections first (behind nodes)
        self._draw_enhanced_connections(projected_points, state)

        # Draw enhanced nodes
        self._draw_enhanced_nodes(projected_points, state)

        # Draw UI overlays
        self._draw_graph_info(state)

        # Draw border
        pygame.draw.rect(self.surface, COLORS['panel_border'],
                        pygame.Rect(0, 0, GRAPH_PANEL_WIDTH, self.surface.get_height()), 2)

        return self.surface

    def _calculate_projections(self) -> Dict[Any, Tuple[int, int, float]]:
        """Calculate 3D to 2D projections with enhanced depth handling."""
        projected_points = {}
        center_x = self.surface.get_width() / 2
        center_y = self.surface.get_height() / 2

        # Create enhanced rotation matrices
        rot_y = np.array([
            [math.cos(self.rotation_angle), 0, math.sin(self.rotation_angle)],
            [0, 1, 0],
            [-math.sin(self.rotation_angle), 0, math.cos(self.rotation_angle)]
        ])

        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(self.rotation_x), -math.sin(self.rotation_x)],
            [0, math.sin(self.rotation_x), math.cos(self.rotation_x)]
        ])

        # Project all nodes
        for node, pos3d in self.node_positions_3d.items():
            # Apply rotations
            rotated_pos = np.dot(rot_y, pos3d)
            rotated_pos = np.dot(rot_x, rotated_pos)

            # Apply perspective projection
            z = rotated_pos[2] + self.camera_distance
            if z <= 0.1:  # Prevent division by zero
                z = 0.1

            x = (rotated_pos[0] / z) * self.scale + center_x
            y = (rotated_pos[1] / z) * self.scale + center_y

            projected_points[node] = (int(x), int(y), z)

        return projected_points

    def _draw_enhanced_connections(self, projected_points: Dict[Any, Tuple[int, int, float]],
                                 state: Dict[str, Any]):
        """Draw enhanced connections with improved visibility."""
        active_node = state.get('active_node_id')

        # Sort edges by depth (draw farther ones first)
        edges_with_depth = []
        for u, v in self.graph.edges():
            if u in projected_points and v in projected_points:
                avg_depth = (projected_points[u][2] + projected_points[v][2]) / 2
                edges_with_depth.append((u, v, avg_depth))

        edges_with_depth.sort(key=lambda x: x[2])

        # Draw each edge with enhanced styling
        for u, v, depth in edges_with_depth:
            p1 = projected_points[u]
            p2 = projected_points[v]

            # Calculate edge properties
            weight = self.edge_weights.get((u, v), self.edge_weights.get((v, u), 1.0))
            activity = self.edge_activities.get((u, v), self.edge_activities.get((v, u), 0.0))

            # Determine edge appearance
            is_active = u == active_node or v == active_node
            thickness = self._calculate_edge_thickness(weight, activity, is_active, depth)
            color = self._calculate_edge_color(weight, activity, is_active, depth)

            # Draw enhanced edge
            if thickness > 1:
                self._draw_enhanced_edge(p1, p2, color, thickness, activity > 0.5)
            else:
                pygame.draw.line(self.surface, color, (p1[0], p1[1]), (p2[0], p2[1]), 1)

    def _draw_enhanced_edge(self, p1: Tuple[int, int, float], p2: Tuple[int, int, float],
                          color: Tuple[int, int, int], thickness: int, glow: bool = False):
        """Draw a single enhanced edge with optional glow effect."""
        start = (p1[0], p1[1])
        end = (p2[0], p2[1])

        if glow:
            # Draw glow effect
            for i in range(thickness * 2, thickness, -1):
                alpha = int(100 * (1 - (i - thickness) / thickness))
                glow_color = (*color, alpha)

                # Create temporary surface for glow
                temp_surface = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
                pygame.draw.line(temp_surface, glow_color, start, end, i)
                self.surface.blit(temp_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)

        # Draw main edge
        pygame.draw.line(self.surface, color, start, end, thickness)

        # Add directional arrow for strong connections
        if thickness >= 3:
            self._draw_edge_arrow(start, end, color, thickness)

    def _draw_edge_arrow(self, start: Tuple[int, int], end: Tuple[int, int],
                        color: Tuple[int, int, int], thickness: int):
        """Draw a directional arrow on the edge."""
        # Calculate arrow position (75% along the edge)
        arrow_pos = (
            int(start[0] + 0.75 * (end[0] - start[0])),
            int(start[1] + 0.75 * (end[1] - start[1]))
        )

        # Calculate arrow direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)

        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length

            # Arrow size based on thickness
            arrow_size = max(3, thickness)

            # Calculate arrow points
            arrow_points = [
                arrow_pos,
                (arrow_pos[0] - arrow_size * dx + arrow_size * dy * 0.5,
                 arrow_pos[1] - arrow_size * dy - arrow_size * dx * 0.5),
                (arrow_pos[0] - arrow_size * dx - arrow_size * dy * 0.5,
                 arrow_pos[1] - arrow_size * dy + arrow_size * dx * 0.5)
            ]

            pygame.draw.polygon(self.surface, color, arrow_points)

    def _draw_enhanced_nodes(self, projected_points: Dict[Any, Tuple[int, int, float]],
                           state: Dict[str, Any]):
        """Draw enhanced nodes with beautiful styling."""
        active_node = state.get('active_node_id')

        # Sort nodes by depth (draw farther ones first)
        nodes_with_depth = [(node, projected_points[node]) for node in self.graph.nodes()
                           if node in projected_points]
        nodes_with_depth.sort(key=lambda x: x[1][2], reverse=True)

        # Draw each node
        for node, (px, py, z) in nodes_with_depth:
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'sensory')
            activation = node_data.get('activation', 0.0)

            # Calculate node appearance
            size = self._calculate_node_size_3d(node_type, activation, z, node == active_node)
            color = self._calculate_node_color(node_type, activation, z, node == active_node)

            # Draw node with enhanced effects
            if node == active_node:
                # Draw pulsing glow for active node
                glow_size = int(size * 2 + 5 * math.sin(self.frame_count * 0.2))
                draw_glowing_circle(self.surface, color, (px, py), size, glow_size)
            else:
                # Draw regular node
                draw_glowing_circle(self.surface, color, (px, py), size, size + 3)

            # Draw node label for important nodes
            if size > 8 or node == active_node:
                self._draw_node_label(node, px, py, size, color)

    def _draw_node_label(self, node: Any, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw node label with enhanced styling."""
        font = get_font(SMALL_FONT_SIZE)
        label = str(node)

        # Create label with shadow
        label_surface = create_text_with_shadow(label, font, COLORS['text_primary'],
                                              (0, 0, 0), (1, 1))

        # Position label below node
        label_rect = label_surface.get_rect(center=(x, y + size + 10))
        self.surface.blit(label_surface, label_rect)

    def _calculate_node_size(self, activation: float) -> int:
        """Calculate node size based on activation."""
        base_size = GRAPH_3D_CONFIG['node_base_size']
        variation = GRAPH_3D_CONFIG['node_size_variation']
        return int(base_size + activation * variation)

    def _calculate_node_size_3d(self, node_type: str, activation: float, depth: float,
                               is_active: bool) -> int:
        """Calculate 3D node size with depth and type considerations."""
        base_size = GRAPH_3D_CONFIG['node_base_size']

        # Type-based size
        if node_type == 'action':
            base_size += 3

        # Activation-based size
        activation_bonus = int(activation * GRAPH_3D_CONFIG['node_size_variation'])

        # Active node bonus
        active_bonus = 8 if is_active else 0

        # Depth-based scaling
        depth_scale = max(0.3, 1.0 / depth)

        final_size = int((base_size + activation_bonus + active_bonus) * depth_scale)
        return max(2, final_size)

    def _calculate_node_color(self, node_type: str, activation: float, depth: float,
                            is_active: bool) -> Tuple[int, int, int]:
        """Calculate node color with enhanced effects."""
        if is_active:
            base_color = COLORS['node_active']
            # Add pulsing effect
            base_color = pulse_color(base_color, self.frame_count, PULSE_SPEED, 0.3)
        elif node_type == 'action':
            base_color = COLORS['node_action']
        else:
            base_color = COLORS['node_sensory']

        # Apply activation intensity
        if activation > 0.1:
            intensity = min(1.0, activation * 2)
            base_color = tuple(int(c + (255 - c) * intensity * 0.3) for c in base_color)

        # Apply depth fading
        return get_depth_color(base_color, depth, 3.0)

    def _calculate_edge_thickness(self, weight: float, activity: float, is_active: bool,
                                depth: float) -> int:
        """Calculate edge thickness based on properties."""
        base_thickness = GRAPH_3D_CONFIG['edge_thickness']

        # Weight-based thickness
        weight_factor = min(2.0, abs(weight))
        thickness = int(base_thickness * weight_factor)

        # Activity bonus
        if activity > 0.5:
            thickness += 1

        # Active connection bonus
        if is_active:
            thickness += GRAPH_3D_CONFIG['edge_thickness_active'] - base_thickness

        # Depth scaling
        depth_scale = max(0.5, 1.0 / depth)
        thickness = int(thickness * depth_scale)

        return max(1, thickness)

    def _calculate_edge_color(self, weight: float, activity: float, is_active: bool,
                            depth: float) -> Tuple[int, int, int]:
        """Calculate edge color with enhanced visibility."""
        if is_active:
            base_color = COLORS['edge_active']
        elif activity > 0.7:
            base_color = COLORS['edge_strong']
        else:
            base_color = COLORS['edge_default']

        # Apply weight intensity
        if abs(weight) > 1.0:
            intensity = min(1.0, abs(weight) / 2.0)
            base_color = tuple(int(c + (255 - c) * intensity * 0.2) for c in base_color)

        # Apply activity pulsing
        if activity > 0.5:
            base_color = pulse_color(base_color, self.frame_count, PULSE_SPEED * 2, 0.2)

        # Apply depth fading
        return get_depth_color(base_color, depth, 3.0)

    def _draw_graph_info(self, state: Dict[str, Any]):
        """Draw graph information overlay."""
        font = get_font(SMALL_FONT_SIZE)

        info_lines = [
            f"Nodes: {self.graph.number_of_nodes()}",
            f"Edges: {self.graph.number_of_edges()}",
            f"Rotation: {'Paused' if self.paused else 'Active'}",
        ]

        y_offset = 10
        for line in info_lines:
            text_surface = create_text_with_shadow(line, font, COLORS['text_secondary'])
            self.surface.blit(text_surface, (10, y_offset))
            y_offset += 15

    def _draw_empty_state(self):
        """Draw empty state message."""
        font = get_font(LARGE_FONT_SIZE)
        text = "No network data"
        text_surface = create_text_with_shadow(text, font, COLORS['text_muted'])

        text_rect = text_surface.get_rect(center=(GRAPH_PANEL_WIDTH // 2,
                                                self.surface.get_height() // 2))
        self.surface.blit(text_surface, text_rect)

        # Draw instructions
        small_font = get_font(SMALL_FONT_SIZE)
        instructions = [
            "Controls:",
            "SPACE - Pause rotation",
            "R - Reset rotation",
            "ESC - Exit"
        ]

        y_offset = text_rect.bottom + 30
        for instruction in instructions:
            inst_surface = create_text_with_shadow(instruction, small_font, COLORS['text_secondary'])
            inst_rect = inst_surface.get_rect(center=(GRAPH_PANEL_WIDTH // 2, y_offset))
            self.surface.blit(inst_surface, inst_rect)
            y_offset += 20
