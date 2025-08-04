#!/usr/bin/env python3
"""
Standalone script to run the STAG visualizer without Django.
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from .visualizer import Visualizer
import .config


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description='Enhanced STAG Agent & Network Visualizer'
    )

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

    args = parser.parse_args()

    # Validate arguments
    if args.size < 5 or args.size > 50:
        print("❌ Error: Grid size must be between 5 and 50.")
        sys.exit(1)

    if args.fps < 10 or args.fps > 120:
        print("❌ Error: FPS must be between 10 and 120.")
        sys.exit(1)

    # Update global FPS setting
    config.FPS = args.fps

    websocket_uri = f"ws://{args.host}:{args.port}/ws/brain/training/{args.network_id}/"

    # Display startup information
    print("🚀 Starting Enhanced STAG Visualizer")
    print(f"🎯 Network ID: {args.network_id}")
    print(f"📏 Grid size: {args.size}x{args.size}")
    print(f"🎬 Target FPS: {args.fps}")
    print(f"📡 WebSocket: {websocket_uri}")
    print("⌨️  Controls: ESC=Quit, SPACE=Pause, R=Reset")
    print("-" * 50)

    try:
        # Check dependencies
        try:
            import pygame
            import numpy as np
            import networkx as nx
            import websockets
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Please install required packages:")
            print("pip install pygame numpy networkx websockets")
            sys.exit(1)

        # Initialize and run the visualizer
        visualizer = Visualizer(size=args.size)
        visualizer.run(websocket_uri)

    except KeyboardInterrupt:
        print("\n⏹️  Visualizer interrupted by user")
    except Exception as e:
        print(f"❌ Failed to start visualizer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("👋 Visualizer closed gracefully.")


if __name__ == "__main__":
    main()
