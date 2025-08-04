import time
import random
import argparse
import os
import sys
from typing import Dict, List, Optional, Callable

# Add the project root to the Python path to allow for Django setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Django Setup ---
# This must be configured before importing any Django models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
from django.utils import timezone

django.setup()
# --- End Django Setup ---

from .environments import EnvironmentManager, BaseEnvironment
from .models import BrainNetwork, TrainingSession
from .services import STAGNetworkService


class GameTrainer:
    """
    Acts as a "Coach" to orchestrate training and evaluation sessions for a STAG network.
    This class does not contain any learning logic itself. It is a client that
    delegates all brain functions to the STAGNetworkService and records the results.
    """

    def __init__(self, network_id: int):
        """
        Initializes the GameTrainer for a specific BrainNetwork.

        Args:
            network_id (int): The primary key of the BrainNetwork to be trained.

        Raises:
            BrainNetwork.DoesNotExist: If the network with the given ID is not found.
        """
        self.network_id = network_id
        # Eagerly load the network model to ensure it exists.
        self.network_model = BrainNetwork.objects.get(id=self.network_id)
        self.env_manager = EnvironmentManager()

        # The service is now loaded on-demand by the methods that need it.
        # This avoids keeping a potentially stale service instance in memory.
        print(f"GameTrainer initialized for network: '{self.network_model.name}' (ID: {self.network_id})")

    def _get_service(self) -> STAGNetworkService:
        """
        Lazy-loads and returns the STAGNetworkService instance.
        This ensures the service is always fresh for each operation.
        """
        # In a real-world scenario, a more sophisticated service cache might be used.
        # For this refactoring, creating a new instance is clean and safe.
        return STAGNetworkService(self.network_id)

    def train(self, environment_name: str, episodes: int, max_steps_per_episode: int = 200) -> TrainingSession:
        """
        Runs a full training session for the network on a given environment.

        This method creates a TrainingSession record, invokes the STAG service to
        run the training, and then saves the final results to the database.

        Args:
            environment_name (str): The name of the environment to train on.
            episodes (int): The number of episodes to run.
            max_steps_per_episode (int): The maximum steps allowed per episode.

        Returns:
            TrainingSession: The completed and saved training session object.
        """
        print(f"Starting training session on '{environment_name}' for {episodes} episodes...")
        service = self._get_service()
        env = self.env_manager.get_env(environment_name)

        session = TrainingSession.objects.create(
            network=self.network_model,
            environment_name=environment_name,
            parameters={'episodes': episodes, 'max_steps': max_steps_per_episode},
            status=TrainingSession.Status.RUNNING
        )

        try:
            # The train_on_env method from the service is a generator.
            # We consume it fully to execute the training.
            training_results = list(service.train_on_env(env, episodes, max_steps_per_episode))

            session.status = TrainingSession.Status.COMPLETED
            session.results = {"episodes": training_results}
            print("Training session completed successfully.")

        except Exception as e:
            session.status = TrainingSession.Status.FAILED
            session.results = {"error": str(e)}
            print(f"ERROR: Training session failed: {e}")
            # Re-raise the exception so the caller (e.g., a view) can handle it.
            raise e

        finally:
            # Ensure the session is always saved, even on failure.
            # FIX: Use timezone.now() to get a datetime object for the DateTimeField.
            session.end_time = timezone.now()
            session.save()

        return session

    def evaluate(self, environment_name: str, episodes: int, max_steps_per_episode: int = 200) -> Dict:
        """
        Evaluates the network's current performance without any learning.

        Args:
            environment_name (str): The name of the environment for evaluation.
            episodes (int): The number of episodes to run for the evaluation.
            max_steps_per_episode (int): The maximum steps allowed per episode.

        Returns:
            Dict: A dictionary containing aggregated evaluation results.
        """
        print(f"Starting evaluation on '{environment_name}' for {episodes} episodes...")
        service = self._get_service()
        env = self.env_manager.get_env(environment_name)

        # The service should ideally have a dedicated evaluation method.
        # For now, we can simulate it by running train_on_env but telling the service
        # not to be in a training state (though the current service refactor handles this).
        # A future improvement would be `service.evaluate_on_env(...)`.

        # For this implementation, we assume the service's `train_on_env` can be
        # run without causing learning if `is_training` is False.
        service.is_training = False  # Explicitly set to evaluation mode

        all_rewards = []
        success_count = 0

        for _ in range(episodes):
            state = env.reset().observation
            total_reward = 0.0
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode:
                # This is a simplified evaluation loop. A real one would be in the service.
                # Here, we just get an action and step the environment.
                # The service would need a `predict_action` method.
                # For now, we'll assume a placeholder logic.

                # Placeholder: This logic should be moved into a service.predict_action() method
                sdr = service.control_learner.visual_cortex.to_sdr(state, steps)
                winners = service.sdr_index.search(sdr, 1, service.graph, service.sdr_helper)
                if not winners:
                    action_name = random.choice(env.action_space)
                else:
                    state_node_id = winners[0][0]
                    action_name = service.control_learner.motor_cortex.select_action(state_node_id, service.graph,
                                                                                     service.action_q_table)

                env_state = env.step(action_name)
                state, reward, done = env_state.observation, env_state.reward, env_state.done
                total_reward += reward
                steps += 1

            all_rewards.append(total_reward)
            if total_reward > 0:  # Simple success metric
                success_count += 1

        service.is_training = True  # Reset service state

        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        success_rate = success_count / episodes if episodes > 0 else 0

        print(f"Evaluation complete. Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2%}")
        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "all_rewards": all_rewards
        }


def main():
    """
    Main function to run the GameTrainer from the command line.
    """
    parser = argparse.ArgumentParser(description="STAG Game Trainer CLI")
    parser.add_argument('--network-id', type=int, required=True, help='The ID of the BrainNetwork to train.')
    parser.add_argument('--action', type=str, default='train', choices=['train', 'evaluate'],
                        help='The action to perform.')
    parser.add_argument('--env', type=str, default='gridworld',
                        help='The environment to use for training or evaluation.')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run.')
    parser.add_argument('--steps', type=int, default=200, help='Maximum steps per episode.')

    args = parser.parse_args()

    try:
        trainer = GameTrainer(network_id=args.network_id)

        if args.action == 'train':
            session = trainer.train(
                environment_name=args.env,
                episodes=args.episodes,
                max_steps_per_episode=args.steps
            )
            print("\n--- Training Session Summary ---")
            print(f"  Session ID: {session.id}")
            print(f"  Status: {session.status}")
            if session.status == TrainingSession.Status.COMPLETED:
                rewards = [e.get('reward', 0) for e in session.results.get('episodes', [])]
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                print(f"  Average Reward: {avg_reward:.2f}")
            else:
                print(f"  Error: {session.results.get('error', 'Unknown error')}")

        elif args.action == 'evaluate':
            results = trainer.evaluate(
                environment_name=args.env,
                episodes=args.episodes,
                max_steps_per_episode=args.steps
            )
            print("\n--- Evaluation Summary ---")
            print(f"  Average Reward: {results['avg_reward']:.2f}")
            print(f"  Success Rate: {results['success_rate']:.2%}")

    except BrainNetwork.DoesNotExist:
        print(f"Error: No BrainNetwork found with ID {args.network_id}.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # This block allows the script to be run from the command line.
    # You can run this via `python -m brain.game_trainer --network-id 1`
    # or by making the file executable (`chmod +x game_trainer.py`) and running `./game_trainer.py --network-id 1`
    main()
