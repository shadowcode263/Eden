"""
Configuration file for the STAG Visualizer with enhanced styling and constants.
"""

# --- Display Constants ---
CELL_SIZE = 45
INFO_PANEL_HEIGHT = 100
PADDING = 10
FONT_SIZE = 16
SMALL_FONT_SIZE = 12
LARGE_FONT_SIZE = 20

# Panel dimensions
GRID_PANEL_WIDTH = 11 * CELL_SIZE + (PADDING * 2)
GRAPH_PANEL_WIDTH = 500
WINDOW_WIDTH = GRID_PANEL_WIDTH + GRAPH_PANEL_WIDTH + (PADDING * 3)

# Animation settings
FPS = 60
ROTATION_SPEED = 0.008
PULSE_SPEED = 0.05

# --- Enhanced Color Palette ---
COLORS = {
    # Background colors
    'bg_top': (25, 30, 40),
    'bg_bottom': (15, 20, 30),
    'panel_bg': (35, 40, 50),
    'panel_border': (60, 70, 85),
    'shadow': (0, 0, 0, 50),

    # Primary colors
    'primary': (64, 150, 255),
    'secondary': (150, 64, 255),
    'accent': (255, 150, 64),
    'success': (64, 255, 150),
    'warning': (255, 200, 64),
    'danger': (255, 64, 100),

    # Text colors
    'text_primary': (240, 245, 250),
    'text_secondary': (180, 190, 200),
    'text_muted': (120, 130, 140),

    # Entity colors (enhanced)
    'empty': (45, 52, 64),
    'player': (64, 150, 255),
    'goal': (64, 255, 150),
    'obstacle': (255, 64, 100),

    # Network colors
    'node_sensory': (100, 200, 255),
    'node_action': (255, 150, 100),
    'node_active': (255, 100, 150),
    'edge_default': (100, 120, 140),
    'edge_active': (255, 200, 100),
    'edge_strong': (150, 255, 150),

    # UI elements
    'border_light': (80, 90, 105),
    'border_dark': (20, 25, 35),
    'highlight': (255, 255, 255, 30),
}

# Entity type mapping (assuming EntityType enum exists)
try:
    from brain.environments import EntityType
    ENTITY_COLOR_MAP = {
        EntityType.EMPTY.value: COLORS['empty'],
        EntityType.PLAYER.value: COLORS['player'],
        EntityType.GOAL.value: COLORS['goal'],
        EntityType.OBSTACLE.value: COLORS['obstacle'],
    }
except ImportError:
    # Fallback mapping
    ENTITY_COLOR_MAP = {
        0: COLORS['empty'],
        1: COLORS['player'],
        2: COLORS['goal'],
        3: COLORS['obstacle'],
    }

# --- 3D Visualization Settings ---
GRAPH_3D_CONFIG = {
    'scale': 180,
    'camera_distance': 2.5,
    'rotation_x': 0.3,
    'node_base_size': 8,
    'node_size_variation': 12,
    'edge_thickness': 2,
    'edge_thickness_active': 4,
    'depth_fade_factor': 0.3,
}
