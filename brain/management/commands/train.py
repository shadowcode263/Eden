from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
import numpy as np

from brain.models import BrainNetwork, TrainingSession
from brain.services import STAGNetworkService
from brain.environments import EnvironmentManager, BaseEnvironment


class Command(BaseCommand):
    """
    A Django management command to run the STAG Game Trainer.
    """
    help = 'Runs the STAG Game Trainer for a specified Brain Network.'

    def add_arguments(self, parser):
        parser.add_argument('--network-id', type=int, required=True, help='The ID of the BrainNetwork to train.')
        parser.add_argument('--action', type=str, default='train', choices=['train', 'evaluate'],
                            help='The action to perform.')
        parser.add_argument('--env', type=str, default='gridworld', help='The environment to use.')
        parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run.')
        parser.add_argument('--steps', type=int, default=200, help='Maximum steps per episode.')

    def handle(self, *args, **options):
        network_id = options['network_id']
        action = options['action']
        env_name = options['env']
        episodes = options['episodes']
        steps = options['steps']

        try:
            self.stdout.write(self.style.SUCCESS(f"Initializing service for network ID: {network_id}..."))
            service = STAGNetworkService(network_id=network_id)
            env = EnvironmentManager.get_env(env_name, size=11) # Ensure size matches visualizer

            if action == 'train':
                self.handle_training(service, env, episodes, steps)
            elif action == 'evaluate':
                self.handle_evaluation(service, env, episodes, steps)

        except BrainNetwork.DoesNotExist:
            raise CommandError(f"Error: No BrainNetwork found with ID {network_id}.")
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"An unexpected error occurred: {e}"))
            import traceback
            traceback.print_exc()


    def handle_training(self, service: STAGNetworkService, env: BaseEnvironment, episodes: int, steps: int):
        self.stdout.write(self.style.SUCCESS(f"Starting training on '{env.name}' for {episodes} episodes..."))

        session = TrainingSession.objects.create(
            network=service.network_model,
            environment_name=env.name,
            parameters={'episodes': episodes, 'max_steps': steps},
            status=TrainingSession.Status.RUNNING
        )

        all_results = []
        try:
            training_generator = service.train_on_env(env, episodes, steps)

            for result in training_generator:
                if result.get('type') == 'episode_end':
                    all_results.append(result)
                    # REFACTORED: Print more detailed end-of-episode stats
                    num_nodes = service.graph.number_of_nodes()
                    num_edges = service.graph.number_of_edges()
                    self.stdout.write(
                        f"  Episode {result['episode']} | "
                        f"Reward: {result['reward']:.2f} | "
                        f"Steps: {result['steps']} | "
                        f"Epsilon: {result['epsilon']:.4f} | "
                        f"Graph: {num_nodes} nodes, {num_edges} edges"
                    )

            session.status = TrainingSession.Status.COMPLETED
            session.results = {"episodes": all_results}
            self.stdout.write(self.style.SUCCESS("\n--- Training Session Summary ---"))
            self.stdout.write(f"  Session ID: {session.id}")
            self.stdout.write(f"  Status: {session.status}")
            rewards = [e['reward'] for e in all_results if e.get('type') == 'episode_end']
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            self.stdout.write(f"  Average Reward: {avg_reward:.2f}")

            # REFACTORED: Add final graph stats to the summary
            final_nodes = service.graph.number_of_nodes()
            final_edges = service.graph.number_of_edges()
            self.stdout.write(self.style.SUCCESS("\n--- Final Graph State ---"))
            self.stdout.write(f"  Total Nodes: {final_nodes}")
            self.stdout.write(f"  Total Edges: {final_edges}")


        except Exception as e:
            session.status = TrainingSession.Status.FAILED
            session.results = {"error": str(e)}
            self.stderr.write(self.style.ERROR(f"Training failed: {e}"))
        finally:
            session.end_time = timezone.now()
            session.save()

    def handle_evaluation(self, service: STAGNetworkService, env: BaseEnvironment, episodes: int, steps: int):
        self.stdout.write(self.style.SUCCESS(f"Starting evaluation on '{env.name}' for {episodes} episodes..."))

        service.is_training = False
        all_rewards = []

        try:
            eval_generator = service.train_on_env(env, episodes, steps)
            for result in eval_generator:
                if result.get('type') == 'episode_end':
                    all_rewards.append(result['reward'])

            avg_reward = np.mean(all_rewards) if all_rewards else 0
            success_rate = np.mean([r > 0 for r in all_rewards]) if all_rewards else 0

            self.stdout.write(self.style.SUCCESS("\n--- Evaluation Summary ---"))
            self.stdout.write(f"  Average Reward: {avg_reward:.2f}")
            self.stdout.write(f"  Success Rate: {success_rate:.2%}")

        finally:
            service.is_training = True
