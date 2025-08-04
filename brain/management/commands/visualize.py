"""
Django management command for the enhanced STAG visualizer.
"""

import sys
import os
from django.core.management.base import BaseCommand, CommandError

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

try:
    from .scripts.visualizer import Visualizer
    from .scripts.config import FPS
except ImportError as e:
    raise CommandError(f"Failed to import visualizer modules: {e}")


class Command(BaseCommand):
    """Enhanced Django management command for the STAG visualizer."""

    help = 'Launches a beautiful GUI to visualize agent training and network graph with enhanced connection visibility.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--network-id',
            type=int,
            required=True,
            help='The ID of the BrainNetwork to visualize.'
        )
        parser.add_argument(
            '--size',
            type=int,
            default=11,
            help='The size of the game grid to display (default: 11).'
        )
        parser.add_argument(
            '--fps',
            type=int,
            default=60,
            help='Target FPS for the visualizer (default: 60).'
        )
        parser.add_argument(
            '--host',
            type=str,
            default='localhost',
            help='WebSocket host (default: localhost).'
        )
        parser.add_argument(
            '--port',
            type=int,
            default=8000,
            help='WebSocket port (default: 8000).'
        )

    def handle(self, *args, **options):
        """Handle the management command execution."""
        network_id = options['network_id']
        size = options['size']
        fps = options.get('fps', 60)
        host = options.get('host', 'localhost')
        port = options.get('port', 8000)

        # Validate arguments
        if size < 5 or size > 50:
            raise CommandError("Grid size must be between 5 and 50.")

        if fps < 10 or fps > 120:
            raise CommandError("FPS must be between 10 and 120.")

        # Update global FPS setting
        import config
        config.FPS = fps

        websocket_uri = f"ws://{host}:{port}/ws/brain/training/{network_id}/"

        # Display startup information
        self.stdout.write(
            self.style.SUCCESS("🚀 Starting Enhanced STAG Visualizer")
        )
        self.stdout.write(f"🎯 Network ID: {network_id}")
        self.stdout.write(f"📏 Grid size: {size}x{size}")
        self.stdout.write(f"🎬 Target FPS: {fps}")
        self.stdout.write(f"📡 WebSocket: {websocket_uri}")
        self.stdout.write(
            self.style.WARNING("Controls: ESC=Quit, SPACE=Pause, R=Reset")
        )

        try:
            # Check if pygame is available
            try:
                import pygame
            except ImportError:
                raise CommandError(
                    "Pygame is not installed. Please install it with: pip install pygame"
                )

            # Check if required modules are available
            try:
                import numpy as np
                import networkx as nx
                import websockets
            except ImportError as e:
                raise CommandError(f"Required dependency missing: {e}")

            # Initialize and run the visualizer
            visualizer = Visualizer(size=size)
            visualizer.run(websocket_uri)

        except KeyboardInterrupt:
            self.stdout.write(
                self.style.SUCCESS("\n⏹️  Visualizer interrupted by user")
            )
        except Exception as e:
            self.stderr.write(
                self.style.ERROR(f"❌ Failed to start visualizer: {e}")
            )
            if options.get('verbosity', 1) >= 2:
                import traceback
                traceback.print_exc()
            raise CommandError(f"Visualizer failed: {e}")
        finally:
            self.stdout.write(
                self.style.SUCCESS("👋 Visualizer closed gracefully.")
            )
