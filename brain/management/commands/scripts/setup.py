#!/usr/bin/env python3
"""
Setup script for the STAG Visualizer.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    requirements_file = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')

    if os.path.exists(requirements_file):
        print("📦 Installing requirements...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', requirements_file
            ])
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    else:
        print("⚠️  Requirements file not found. Installing core dependencies...")
        core_deps = ['pygame', 'numpy', 'networkx', 'websockets']
        for dep in core_deps:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                print(f"✅ Installed {dep}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {dep}: {e}")
                return False

    return True


def check_dependencies():
    """Check if all dependencies are available."""
    dependencies = {
        'pygame': 'Pygame (graphics library)',
        'numpy': 'NumPy (numerical computing)',
        'networkx': 'NetworkX (graph library)',
        'websockets': 'WebSockets (real-time communication)',
    }

    missing = []
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - MISSING")
            missing.append(module)

    return len(missing) == 0


def main():
    """Main setup function."""
    print("🔧 STAG Visualizer Setup")
    print("=" * 30)

    print("\n📋 Checking dependencies...")
    if check_dependencies():
        print("\n🎉 All dependencies are available!")
        print("\nYou can now run the visualizer:")
        print("  Django: python manage.py visualize_training --network-id 1")
        print("  Standalone: python scripts/run_visualizer.py --network-id 1")
    else:
        print("\n📦 Some dependencies are missing.")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if install_requirements():
                print("\n🎉 Setup complete!")
            else:
                print("\n❌ Setup failed. Please install dependencies manually.")
                sys.exit(1)
        else:
            print("Please install the missing dependencies manually.")
            sys.exit(1)


if __name__ == "__main__":
    main()
