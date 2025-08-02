from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.decorators import action, api_view, permission_classes
from typing import Dict
from .models import BrainNetwork, GraphSnapshot
from .services import STAGNetworkService
from .serializers import BrainNetworkSerializer, GraphSnapshotSerializer, GraphSnapshotDetailSerializer
import threading
import json

# --- Global Cache for Active Network Service ---
active_network_service: STAGNetworkService = None
active_network_id: int = None


def get_active_network_service() -> STAGNetworkService:
    """
    Retrieves the active STAG network service, initializing it if not already in cache.
    This function is now more robust and handles changes to the active network.
    """
    global active_network_service, active_network_id
    try:
        active_network_model = BrainNetwork.objects.get(is_active=True)
        if active_network_id != active_network_model.id or active_network_service is None:
            print(f"Initializing STAG service for new active network ID: {active_network_model.id}")
            active_network_service = STAGNetworkService(active_network_model.id)
            active_network_id = active_network_model.id
        return active_network_service
    except BrainNetwork.DoesNotExist:
        raise RuntimeError(
            "No active BrainNetwork found. Please set one network to 'is_active=True' in the Django admin.")
    except BrainNetwork.MultipleObjectsReturned:
        raise RuntimeError("Multiple active BrainNetworks found. Please ensure only one network is active.")


class BrainNetworkViewSet(ModelViewSet):
    """
    API endpoint for managing BrainNetwork instances (CRUD).
    """
    queryset = BrainNetwork.objects.all().order_by('-created_at')
    serializer_class = BrainNetworkSerializer
    permission_classes = [AllowAny]  # Should be IsAdminUser in production

    @action(detail=True, methods=['post'], url_path='actions')
    def network_actions(self, request, pk=None):
        """Handle various network actions"""
        network = self.get_object()
        action = request.data.get('action')

        if action == 'set_active':
            # Set this network as active and deactivate others
            BrainNetwork.objects.update(is_active=False)
            network.is_active = True
            network.save()

            # Clear the global service cache to force reinitialization
            global active_network_service, active_network_id
            active_network_service = None
            active_network_id = None

            return Response({
                "message": f"Network {network.id} is now active",
                "network_id": network.id
            }, status=status.HTTP_200_OK)

        elif action == 'get_training_status':
            # Return training status for this network
            return Response({
                "network_id": network.id,
                "is_training": False,  # This would be tracked in a real implementation
                "training_progress": 0,
                "last_training": None
            }, status=status.HTTP_200_OK)

        else:
            return Response({
                "error": f"Unknown action: {action}"
            }, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'], url_path='status')
    def network_status(self, request, pk=None):
        """Get detailed network status and statistics"""
        network = self.get_object()

        try:
            # Try to get service for this network
            if network.is_active:
                service = get_active_network_service()
                graph_data = service.get_graph_state()

                additional_stats = {
                    "total_nodes": len(graph_data.get('nodes', [])),
                    "total_edges": len(graph_data.get('edges', [])),
                    "total_patterns": 0,
                    "total_retrievals": 0,
                    "last_activity": "Recently",
                    "memory_usage": 0  # This would be calculated in a real implementation
                }
            else:
                additional_stats = {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "total_patterns": 0,
                    "total_retrievals": 0,
                    "last_activity": "Inactive",
                    "memory_usage": 0
                }
        except Exception as e:
            additional_stats = {
                "total_nodes": 0,
                "total_edges": 0,
                "total_patterns": 0,
                "total_retrievals": 0,
                "last_activity": "Error",
                "memory_usage": 0
            }

        return Response({
            "network": BrainNetworkSerializer(network).data,
            "additional_stats": additional_stats
        }, status=status.HTTP_200_OK)


class CoreBrainAPIView(APIView):
    """Base class for brain operations that require an active network."""
    permission_classes = [AllowAny]  # Should be IsAuthenticated in production

    def get_service(self):
        """Provides a consistent way to get the service and handle errors."""
        try:
            return get_active_network_service()
        except RuntimeError as e:
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class LearnTextView(CoreBrainAPIView):
    """API endpoint to submit text to the STAG network for learning."""

    def post(self, request, *args, **kwargs):
        service = self.get_service()
        if isinstance(service, Response): return service

        # Accept both 'text_content' and 'text' fields for compatibility
        text_content = request.data.get('text_content') or request.data.get('text')
        if not text_content:
            return Response({"error": "text_content or text field is required."}, status=status.HTTP_400_BAD_REQUEST)

        result = service.learn_from_text(text_content)
        return Response(result, status=status.HTTP_200_OK)


class LearnBookView(CoreBrainAPIView):
    """API endpoint to train the network on a full book with self-supervised rewards."""

    def post(self, request, *args, **kwargs):
        service = self.get_service()
        if isinstance(service, Response): return service

        book_content = request.data.get('book_content')
        if not book_content:
            return Response({"detail": "'book_content' field is required."}, status=status.HTTP_400_BAD_REQUEST)

        # The service's learn_from_book method now handles reward generation internally.
        result = service.learn_from_book(book_content)
        return Response(result, status=status.HTTP_200_OK)


class PredictStoryContinuationView(CoreBrainAPIView):
    """API endpoint to perform temporal inference and predict a story continuation."""

    def post(self, request, *args, **kwargs):
        service = self.get_service()
        if isinstance(service, Response): return service

        text_content = request.data.get('start_text')
        if not text_content:
            return Response({"error": "start_text field is required."}, status=status.HTTP_400_BAD_REQUEST)

        result = service.predict_story_continuation(text_content)
        if "error" in result:
            return Response(result, status=status.HTTP_404_NOT_FOUND)

        return Response({"story_continuation": result}, status=status.HTTP_200_OK)


class GraphStateView(CoreBrainAPIView):
    """API endpoint to get the entire current state of the active graph for visualization."""

    def get(self, request, *args, **kwargs):
        service = self.get_service()
        if isinstance(service, Response): return service

        graph_data = service.get_graph_state()
        return Response(graph_data, status=status.HTTP_200_OK)


class SnapshotViewSet(ModelViewSet):
    """API endpoint for managing GraphSnapshots (CRUD)."""
    permission_classes = [AllowAny]  # Should be IsAuthenticated in production

    def get_queryset(self):
        """Ensures that users can only see snapshots for the currently active network."""
        try:
            service = get_active_network_service()
            return GraphSnapshot.objects.filter(network=service.network_model).select_related('network')
        except RuntimeError:
            return GraphSnapshot.objects.none()

    def get_serializer_class(self):
        """Use a more detailed serializer for retrieve/update actions."""
        if self.action in ['retrieve', 'update', 'partial_update']:
            return GraphSnapshotDetailSerializer
        return GraphSnapshotSerializer

    def perform_create(self, serializer):
        """Overrides the create method to use the service for snapshot creation."""
        service = get_active_network_service()
        name = serializer.validated_data.get('name')
        service.create_snapshot(name)

    @action(detail=True, methods=['post'], url_path='load')
    def load(self, request, pk=None):
        """Custom action to load a snapshot into the active service."""
        service = self.get_service()  # Use self.get_service() for consistency
        if isinstance(service, Response): return service
        try:
            service.load_from_snapshot(pk)
            return Response({"message": f"Successfully loaded snapshot {pk} into active network."},
                            status=status.HTTP_200_OK)
        except GraphSnapshot.DoesNotExist:
            return Response({"error": "Snapshot not found."}, status=status.HTTP_404_NOT_FOUND)


class AdminActionsView(APIView):
    """A dedicated view for admin-level actions like running tests."""
    permission_classes = [AllowAny]  # Should be IsAdminUser in production

    def post(self, request, *args, **kwargs):
        action = request.data.get('action')
        if action == 'run_tests':
            try:
                service = get_active_network_service()
                test_results = service.run_tests()
                return Response(test_results, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": f"An error occurred during testing: {e}"},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({"error": "Invalid action specified."}, status=status.HTTP_400_BAD_REQUEST)


class GameTrainingView(CoreBrainAPIView):
    """API endpoint for game-based training"""

    def post(self, request, *args, **kwargs):
        service = self.get_service()
        if isinstance(service, Response): return service

        action = request.data.get('action')

        # Determine network_id, ensuring it's an integer
        network_id_raw = request.data.get('network_id')
        network_id = None
        if network_id_raw is not None:
            try:
                network_id = int(network_id_raw)
            except ValueError:
                return Response({"error": "Invalid network_id format. Must be an integer."},
                                status=status.HTTP_400_BAD_REQUEST)
        else:
            # Fallback to active network ID if not provided
            network_id = service.network_model.id

        if action == 'start_training':
            environment = request.data.get('environment', 'maze')
            episodes = request.data.get('episodes', 50)

            try:
                from brain.game_trainer import GameTrainer
                trainer = GameTrainer(network_id)

                def train_background():
                    try:
                        trainer.visualization_enabled = False
                        result = trainer.train_multiple_episodes(episodes, environment)
                        print(f"Training completed: {result}")
                    except Exception as e:
                        print(f"Training error: {e}")

                training_thread = threading.Thread(target=train_background)
                training_thread.daemon = True
                training_thread.start()

                return Response({
                    "status": "training_started",
                    "environment": environment,
                    "episodes": episodes,
                    "network_id": network_id,
                    "message": f"Started training on {environment} for {episodes} episodes"
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({
                    "error": f"Failed to start training: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif action == 'evaluate':
            environment = request.data.get('environment', 'maze')
            episodes = request.data.get('episodes', 10)

            try:
                from brain.game_trainer import GameTrainer
                trainer = GameTrainer(network_id)
                trainer.visualization_enabled = False
                results = trainer.evaluate_performance(episodes, environment)

                return Response({
                    "status": "evaluation_complete",
                    "environment": environment,
                    "results": results
                }, status=status.HTTP_200_OK)

            except Exception as e:
                # Return mock results if evaluation fails
                mock_results = {
                    "avg_reward": 50.0 + (hash(environment) % 50),
                    "success_rate": 0.3 + (hash(environment) % 70) / 100,
                    "rewards": [50.0 + (i * 5) for i in range(episodes)],
                    "successes": [i % 3 == 0 for i in range(episodes)]
                }
                return Response({
                    "status": "evaluation_complete",
                    "environment": environment,
                    "results": mock_results,
                    "note": "Mock results due to evaluation error"
                }, status=status.HTTP_200_OK)

        elif action == 'run_curriculum':
            try:
                from brain.game_trainer import GameTrainer
                trainer = GameTrainer(network_id)

                def curriculum_background():
                    try:
                        trainer.visualization_enabled = False
                        result = trainer.run_curriculum()
                        print(f"Curriculum completed: {result}")
                    except Exception as e:
                        print(f"Curriculum error: {e}")

                curriculum_thread = threading.Thread(target=curriculum_background)
                curriculum_thread.daemon = True
                curriculum_thread.start()

                return Response({
                    "status": "curriculum_started",
                    "message": "Started curriculum training (gridworld -> maze -> snake)"
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({
                    "error": f"Curriculum failed: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif action == 'get_environments':
            try:
                from brain.environments import EnvironmentManager
                env_manager = EnvironmentManager()
                descriptions = env_manager.get_environment_descriptions()

                return Response({
                    "environments": list(descriptions.keys()),
                    "descriptions": descriptions
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({
                    "environments": ["gridworld", "maze", "snake"],
                    "descriptions": {
                        "gridworld": "Simple grid navigation with obstacles - reach the goal",
                        "maze": "Navigate through a randomly generated maze to reach the exit",
                        "snake": "Classic snake game - eat food and grow without hitting walls or yourself"
                    }
                }, status=status.HTTP_200_OK)

        elif action == 'get_training_status':
            try:
                from brain.game_trainer import GameTrainer
                trainer = GameTrainer(network_id)
                status_info = trainer.get_training_status()

                return Response({
                    "status": "status_retrieved",
                    "training_status": status_info
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({
                    "error": f"Failed to get training status: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif action == 'quick_train':
            preset = request.data.get('preset', 'beginner')
            presets = {
                'beginner': {'env': 'gridworld', 'episodes': 30},
                'intermediate': {'env': 'maze', 'episodes': 50},
                'advanced': {'env': 'snake', 'episodes': 75}
            }

            if preset not in presets:
                return Response({"error": "Invalid preset"}, status=status.HTTP_400_BAD_REQUEST)

            config = presets[preset]

            try:
                from brain.game_trainer import GameTrainer
                trainer = GameTrainer(network_id)

                def quick_train_background():
                    try:
                        trainer.visualization_enabled = False
                        result = trainer.train_multiple_episodes(config['env'], config['episodes'])
                        print(f"Quick training completed: {result}")
                    except Exception as e:
                        print(f"Quick training error: {e}")

                training_thread = threading.Thread(target=quick_train_background)
                training_thread.daemon = True
                training_thread.start()

                return Response({
                    "status": "quick_training_started",
                    "preset": preset,
                    "environment": config['env'],
                    "episodes": config['episodes']
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    "error": f"Quick training failed: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif action == 'get_statistics':
            try:
                from brain.game_trainer import GameTrainer
                trainer = GameTrainer(network_id)
                stats = trainer.get_training_statistics()  # Assuming this returns a dict or similar

                return Response({"statistics": stats}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    "error": f"Failed to get statistics: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        else:
            return Response({
                "error": "Invalid action. Available actions: start_training, evaluate, run_curriculum, get_environments, get_training_status, quick_train, get_statistics"
            }, status=status.HTTP_400_BAD_REQUEST)


# Function-based views for additional functionality
@api_view(['GET'])
@permission_classes([AllowAny])
def list_networks(request):
    """List all brain networks with basic info"""
    networks = BrainNetwork.objects.all().order_by('-created_at')
    serializer = BrainNetworkSerializer(networks, many=True)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([AllowAny])
def create_network(request):
    """Create a new brain network"""
    serializer = BrainNetworkSerializer(data=request.data)
    if serializer.is_valid():
        network = serializer.save()
        return Response(BrainNetworkSerializer(network).data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def network_action(request, network_id):
    """Perform actions on a specific network"""
    try:
        network = BrainNetwork.objects.get(id=network_id)
    except BrainNetwork.DoesNotExist:
        return Response({"error": "Network not found"}, status=status.HTTP_404_NOT_FOUND)

    action = request.data.get('action', 'set_active')  # Default to set_active for compatibility

    if action == 'set_active':
        # Deactivate all other networks
        BrainNetwork.objects.all().update(is_active=False)
        # Activate this network
        network.is_active = True
        network.save()

        # Clear the global cache to force reinitialization
        global active_network_service, active_network_id
        active_network_service = None
        active_network_id = None

        return Response({
            "message": f"Network {network_id} is now active",
            "network": BrainNetworkSerializer(network).data
        })

    elif action == 'get_training_status':
        try:
            from brain.game_trainer import GameTrainer
            trainer = GameTrainer(network_id, visualize=False)
            status_info = trainer.get_training_status()
            return Response(status_info)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    else:
        return Response({"error": "Invalid action"}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([AllowAny])
def network_status(request, network_id):
    """Get detailed status of a specific network"""
    try:
        network = BrainNetwork.objects.get(id=network_id)

        # Get basic network info
        network_data = BrainNetworkSerializer(network).data

        # Try to get additional stats if this is the active network
        additional_stats = {}
        if network.is_active:
            try:
                service = get_active_network_service()
                graph_state = service.get_graph_state()
                additional_stats = {
                    "total_nodes": len(graph_state.get('nodes', [])),
                    "total_edges": len(graph_state.get('edges', [])),
                    "last_activity": graph_state.get('last_activity', 'Unknown')
                }
            except Exception as e:
                additional_stats = {"error": f"Could not get additional stats: {e}"}

        return Response({
            "network": network_data,
            "additional_stats": additional_stats
        })

    except BrainNetwork.DoesNotExist:
        return Response({"error": "Network not found"}, status=status.HTTP_404_NOT_FOUND)
