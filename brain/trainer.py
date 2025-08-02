import os
import django
import random
import argparse
from typing import List, Dict

# --- Django Setup ---
# This setup is required to run this script in a standalone manner,
# allowing it to access your Django models and services.
# Replace 'eden.settings' with your project's settings file.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "eden.settings")
django.setup()
# --- End Django Setup ---

from brain.services import STAGNetworkService
from brain.models import BrainNetwork


class BookTrainer:
    """
    A robust trainer class for orchestrating the reinforcement learning
    process by feeding book content to the STAG network.
    """

    def __init__(self, network_id: int):
        """
        Initializes the trainer by loading the active STAG network service.

        Args:
            network_id: The ID of the BrainNetwork to be trained.
        """
        print("Initializing Book Trainer...")
        try:
            # Ensure the specified network is set to active for the service to load it.
            BrainNetwork.objects.update(is_active=False)
            network = BrainNetwork.objects.get(pk=network_id)
            network.is_active = True
            network.save()

            self.service = STAGNetworkService(network_id)
            print(f"Successfully connected to Brain Network: '{network.name}' (ID: {network_id})")
        except BrainNetwork.DoesNotExist:
            print(f"Error: BrainNetwork with ID {network_id} does not exist.")
            self.service = None
        except Exception as e:
            print(f"An unexpected error occurred during initialization: {e}")
            self.service = None

    def train_on_book(self, book_path: str, rewards_path: str):
        """
        Trains the network on a single book from a file, applying chapter-based rewards.

        Args:
            book_path: The file path to the book's text content.
            rewards_path: The file path to the chapter rewards (one float per line).
        """
        if not self.service:
            print("Trainer not initialized. Cannot train.")
            return

        try:
            with open(book_path, 'r', encoding='utf-8') as f:
                book_content = f.read()
            with open(rewards_path, 'r', encoding='utf-8') as f:
                chapter_rewards = [float(line.strip()) for line in f if line.strip()]
        except FileNotFoundError as e:
            print(f"Error: Could not find a required file. {e}")
            return
        except ValueError:
            print(f"Error: Rewards file at '{rewards_path}' contains non-numeric values.")
            return

        book_title = os.path.basename(book_path)
        print(f"\n--- Starting Training Session for: '{book_title}' ---")
        result = self.service.learn_from_book(book_content, chapter_rewards)
        print(f"Training complete. Status: {result.get('status')}, Chapters processed: {result.get('chapters')}")

        snapshot_name = f"snapshot_after_{book_title.replace(' ', '_').lower()}"
        print(f"Creating snapshot: '{snapshot_name}'...")
        self.service.create_snapshot(snapshot_name)
        print("Snapshot created successfully.")

    def generate_story(self, prompt: str, max_length: int = 15):
        """
        Uses the trained network to generate a story continuation from a prompt.

        Args:
            prompt: The starting text for the story.
            max_length: The maximum number of concepts (words) in the generated story.
        """
        if not self.service:
            print("Trainer not initialized. Cannot generate story.")
            return

        print(f"\n--- Generating Story from Prompt: '{prompt}' ---")
        concepts = self.service.predict_story_continuation(prompt, max_length)

        if concepts:
            story = ' '.join(concepts)
            print(f"Generated Story: {prompt} {story}...")
        else:
            print("Could not generate a story continuation.")

    def run_tests(self):
        """Runs the built-in test suite for the STAG network service."""
        if not self.service:
            print("Trainer not initialized. Cannot run tests.")
            return
        self.service.run_tests()


def main():
    """
    Main function to parse command-line arguments and run the trainer.
    """
    parser = argparse.ArgumentParser(description="STAG Network Trainer and CLI")
    parser.add_argument('--network-id', type=int, required=True, help="The ID of the BrainNetwork to use.")

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help="Train the network on a book.")
    train_parser.add_argument('--book-path', type=str, required=True, help="Path to the book text file.")
    train_parser.add_argument('--rewards-path', type=str, required=True, help="Path to the chapter rewards file.")

    # Generate command
    generate_parser = subparsers.add_parser('generate', help="Generate a story continuation.")
    generate_parser.add_argument('--prompt', type=str, required=True, help="The starting prompt for the story.")
    generate_parser.add_argument('--length', type=int, default=15, help="Maximum length of the generated story.")

    # Test command
    subparsers.add_parser('test', help="Run the internal test suite.")

    args = parser.parse_args()

    trainer = BookTrainer(args.network_id)

    if trainer.service:
        if args.command == 'train':
            trainer.train_on_book(args.book_path, args.rewards_path)
        elif args.command == 'generate':
            trainer.generate_story(args.prompt, args.length)
        elif args.command == 'test':
            trainer.run_tests()


if __name__ == "__main__":
    # Example usage from the command line:
    # python -m brain.trainer --network-id 1 train --book-path ./sample_book.txt --rewards-path ./sample_rewards.txt
    # python -m brain.trainer --network-id 1 generate --prompt "the hero"
    # python -m brain.trainer --network-id 1 test
    main()
