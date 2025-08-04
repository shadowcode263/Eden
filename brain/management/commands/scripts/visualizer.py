import threading
import queue
import asyncio
import websockets
import json
import numpy as np
import pygame
import math
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from dataclasses import dataclass
from enum import Enum

# Import our modules
from .config import *
from .ui_components import *
from .panels import GridPanel, InfoPanel, GraphPanel


class Visualizer:
    """
    Enhanced STAG Agent & Network Visualizer with beautiful UI and improved connection visibility.
    """

    def __init__(self, size: int = 11):
        pygame.init()
        self.size = size
        self.window_height = (size * CELL_SIZE) + INFO_PANEL_HEIGHT

        # Create main window with enhanced styling
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, self.window_height))
        pygame.display.set_caption("🧠 STAG Agent & Network Visualizer")

        # Set window icon (if available)
        try:
            icon = pygame.Surface((32, 32))
            icon.fill(COLORS['primary'])
            pygame.display.set_icon(icon)
        except:
            pass

        self.clock = pygame.time.Clock()
        self.update_queue = queue.Queue()
        self.latest_state = {}
        self.running = True

        # Initialize enhanced panels
        self.grid_panel = GridPanel(size)
        self.info_panel = InfoPanel()
        self.graph_panel = GraphPanel()

        # Animation state
        self.frame_count = 0

    def run(self, websocket_uri: str):
        """Starts the websocket listener and the main Pygame loop with enhanced error handling."""
        print(f"🚀 Starting STAG Visualizer...")
        print(f"📡 Connecting to: {websocket_uri}")

        # Start WebSocket listener thread
        listener_thread = threading.Thread(
            target=lambda: asyncio.run(self.websocket_listener(websocket_uri)),
            daemon=True
        )
        listener_thread.start()

        # Main game loop with enhanced frame timing
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)

        # Cleanup
        self._cleanup()
        listener_thread.join(timeout=1)

    def _handle_events(self):
        """Handle pygame events with enhanced input processing."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.graph_panel.toggle_pause()
                elif event.key == pygame.K_r:
                    self.graph_panel.reset_rotation()

    def _update(self):
        """Update game state and animations."""
        self.frame_count += 1
        self._check_for_updates()

        # Update panels
        self.grid_panel.update(self.frame_count)
        self.info_panel.update(self.frame_count)
        self.graph_panel.update(self.frame_count)

    def _render(self):
        """Render all components with enhanced visual effects."""
        # Clear screen with gradient background
        self._draw_background()

        # Draw panels with shadows and borders
        self._draw_panel_with_shadow(self.grid_panel, (PADDING, PADDING))
        self._draw_panel_with_shadow(self.info_panel, (PADDING, self.size * CELL_SIZE + PADDING))
        self._draw_panel_with_shadow(self.graph_panel, (GRID_PANEL_WIDTH + PADDING, PADDING))

        # Draw UI overlays
        self._draw_ui_overlays()

        pygame.display.flip()

    def _draw_background(self):
        """Draw an enhanced gradient background."""
        for y in range(self.window_height):
            ratio = y / self.window_height
            color = self._interpolate_color(COLORS['bg_top'], COLORS['bg_bottom'], ratio)
            pygame.draw.line(self.screen, color, (0, y), (WINDOW_WIDTH, y))

    def _draw_panel_with_shadow(self, panel, position):
        """Draw panel with drop shadow effect."""
        shadow_offset = 3
        shadow_color = COLORS['shadow']

        # Draw shadow
        shadow_surface = panel.draw(self.latest_state)
        shadow_surface.fill(shadow_color, special_flags=pygame.BLEND_MULT)
        self.screen.blit(shadow_surface, (position[0] + shadow_offset, position[1] + shadow_offset))

        # Draw actual panel
        panel_surface = panel.draw(self.latest_state)
        self.screen.blit(panel_surface, position)

    def _draw_ui_overlays(self):
        """Draw UI overlays like FPS counter and controls."""
        font = get_font(14)

        # FPS counter
        fps_text = font.render(f"FPS: {int(self.clock.get_fps())}", True, COLORS['text_secondary'])
        self.screen.blit(fps_text, (WINDOW_WIDTH - 80, 10))

        # Controls hint
        controls = ["ESC: Quit", "SPACE: Pause", "R: Reset"]
        for i, control in enumerate(controls):
            text = font.render(control, True, COLORS['text_secondary'])
            self.screen.blit(text, (WINDOW_WIDTH - 120, 30 + i * 15))

    def _check_for_updates(self):
        """Process incoming WebSocket updates."""
        try:
            while not self.update_queue.empty():
                item = self.update_queue.get_nowait()
                if item is None:
                    self.running = False
                    return

                event_type = item.get("event_type")
                if event_type in ["step_update", "activation_update", "graph_update"]:
                    if event_type == "graph_update":
                        self.graph_panel.update_data(item)
                    self.latest_state.update(item)
        except queue.Empty:
            pass

    def _cleanup(self):
        """Clean up resources."""
        pygame.quit()
        self.update_queue.put(None)
        print("👋 Visualizer closed gracefully.")

    @staticmethod
    def _interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float) -> Tuple[
        int, int, int]:
        """Interpolate between two colors."""
        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))

    async def websocket_listener(self, uri: str):
        """Enhanced WebSocket listener with better error handling and reconnection."""
        reconnect_delay = 1
        max_reconnect_delay = 30

        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    print(f"✅ Connected to WebSocket at {uri}")
                    reconnect_delay = 1  # Reset delay on successful connection

                    while self.running:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(message)
                            payload = data.get('payload', data)

                            if payload.get('event_type') in ['step_update', 'activation_update', 'graph_update']:
                                self.update_queue.put(payload)

                        except asyncio.TimeoutError:
                            continue  # Continue listening
                        except websockets.exceptions.ConnectionClosed:
                            print("🔌 WebSocket connection closed")
                            break

            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
                print(f"⚠️  Connection issue: {type(e).__name__}. Retrying in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

            except Exception as e:
                print(f"❌ Unexpected WebSocket error: {e}")
                await asyncio.sleep(reconnect_delay)