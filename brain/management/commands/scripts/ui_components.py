"""
Enhanced UI components and utilities for the STAG Visualizer.
"""

import pygame
import math
from typing import Tuple, Optional
from .config import COLORS, FONT_SIZE, SMALL_FONT_SIZE, LARGE_FONT_SIZE

# Font cache
_font_cache = {}

def get_font(size: int = FONT_SIZE, bold: bool = False) -> pygame.font.Font:
    """Get a cached font object."""
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.Font(None, size)
        if bold:
            _font_cache[key].set_bold(True)
    return _font_cache[key]

def draw_rounded_rect(surface: pygame.Surface, color: Tuple[int, int, int],
                     rect: pygame.Rect, radius: int = 5):
    """Draw a rounded rectangle."""
    if radius <= 0:
        pygame.draw.rect(surface, color, rect)
        return

    # Draw the main rectangle
    inner_rect = pygame.Rect(rect.x + radius, rect.y, rect.width - 2*radius, rect.height)
    pygame.draw.rect(surface, color, inner_rect)

    inner_rect = pygame.Rect(rect.x, rect.y + radius, rect.width, rect.height - 2*radius)
    pygame.draw.rect(surface, color, inner_rect)

    # Draw the corners
    pygame.draw.circle(surface, color, (rect.x + radius, rect.y + radius), radius)
    pygame.draw.circle(surface, color, (rect.x + rect.width - radius, rect.y + radius), radius)
    pygame.draw.circle(surface, color, (rect.x + radius, rect.y + rect.height - radius), radius)
    pygame.draw.circle(surface, color, (rect.x + rect.width - radius, rect.y + rect.height - radius), radius)

def draw_gradient_rect(surface: pygame.Surface, color1: Tuple[int, int, int],
                      color2: Tuple[int, int, int], rect: pygame.Rect, vertical: bool = True):
    """Draw a gradient rectangle."""
    if vertical:
        for y in range(rect.height):
            ratio = y / rect.height
            color = tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))
            pygame.draw.line(surface, color, (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))
    else:
        for x in range(rect.width):
            ratio = x / rect.width
            color = tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))
            pygame.draw.line(surface, color, (rect.x + x, rect.y), (rect.x + x, rect.y + rect.height))

def draw_glowing_circle(surface: pygame.Surface, color: Tuple[int, int, int],
                       center: Tuple[int, int], radius: int, glow_radius: int = None):
    """Draw a circle with a glowing effect."""
    if glow_radius is None:
        glow_radius = radius * 2

    # Draw glow layers
    for i in range(glow_radius, radius, -2):
        alpha = int(255 * (1 - (i - radius) / (glow_radius - radius)) * 0.1)
        glow_color = (*color, alpha)
        glow_surface = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, glow_color, (i, i), i)
        surface.blit(glow_surface, (center[0] - i, center[1] - i), special_flags=pygame.BLEND_ALPHA_SDL2)

    # Draw main circle
    pygame.draw.circle(surface, color, center, radius)

def draw_enhanced_line(surface: pygame.Surface, color: Tuple[int, int, int],
                      start: Tuple[int, int], end: Tuple[int, int],
                      thickness: int = 1, glow: bool = False):
    """Draw an enhanced line with optional glow effect."""
    if glow and thickness > 1:
        # Draw glow effect
        for i in range(thickness * 3, thickness, -1):
            alpha = int(50 * (1 - (i - thickness) / (thickness * 2)))
            glow_color = (*color, alpha)
            # Create a temporary surface for the glow line
            temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.line(temp_surface, glow_color, start, end, i)
            surface.blit(temp_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)

    # Draw main line
    pygame.draw.line(surface, color, start, end, thickness)

def create_text_with_shadow(text: str, font: pygame.font.Font,
                           color: Tuple[int, int, int], shadow_color: Tuple[int, int, int] = None,
                           shadow_offset: Tuple[int, int] = (2, 2)) -> pygame.Surface:
    """Create text with a drop shadow effect."""
    if shadow_color is None:
        shadow_color = (0, 0, 0)

    # Render shadow
    shadow_surface = font.render(text, True, shadow_color)

    # Render main text
    text_surface = font.render(text, True, color)

    # Create combined surface
    width = text_surface.get_width() + abs(shadow_offset[0])
    height = text_surface.get_height() + abs(shadow_offset[1])
    combined_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # Blit shadow first
    shadow_x = max(0, shadow_offset[0])
    shadow_y = max(0, shadow_offset[1])
    combined_surface.blit(shadow_surface, (shadow_x, shadow_y))

    # Blit main text
    text_x = max(0, -shadow_offset[0])
    text_y = max(0, -shadow_offset[1])
    combined_surface.blit(text_surface, (text_x, text_y))

    return combined_surface

def pulse_color(base_color: Tuple[int, int, int], frame: int,
               speed: float = 0.05, intensity: float = 0.3) -> Tuple[int, int, int]:
    """Create a pulsing color effect."""
    pulse = (math.sin(frame * speed) + 1) / 2  # 0 to 1
    factor = 1 + (pulse * intensity)
    return tuple(min(255, int(c * factor)) for c in base_color)

def get_depth_color(base_color: Tuple[int, int, int], depth: float,
                   max_depth: float = 3.0) -> Tuple[int, int, int]:
    """Adjust color based on depth for 3D effect."""
    depth_factor = max(0.2, 1 - (depth / max_depth))
    return tuple(int(c * depth_factor) for c in base_color)
