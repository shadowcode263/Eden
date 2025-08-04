from django.db import transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet, ReadOnlyModelViewSet
from typing import Union

from .models import BrainNetwork, GraphSnapshot, TrainingSession
from .serializers import (
    BrainNetworkDetailSerializer,
    BrainNetworkSerializer,
    GraphSnapshotSerializer,
    TrainingSessionSerializer,
)
from .services import STAGNetworkService


# --- Service Cache ---

class ServiceCache:
    """Manages the lifecycle and caching of STAGNetworkService instances."""
    _cache: dict[int, STAGNetworkService] = {}
    _active_network_id: Union[int, None] = None

    @classmethod
    def get(cls, network_id: int) -> STAGNetworkService:
        """Retrieves a cached service or creates a new one."""
        if network_id not in cls._cache:
            print(f"Cache miss. Initializing service for network_id: {network_id}")
            cls._cache[network_id] = STAGNetworkService(network_id)

        # Update last accessed time for tracking usage
        BrainNetwork.objects.filter(pk=network_id).update(last_accessed=timezone.now())
        return cls._cache[network_id]

    @classmethod
    def set_active(cls, network_id: int):
        """Sets a network as active, ensuring no others are."""
        with transaction.atomic():
            BrainNetwork.objects.exclude(pk=network_id).update(is_active=False)
            BrainNetwork.objects.filter(pk=network_id).update(is_active=True)

        # Update the class variable and clear the specific cache entry
        # to force a reload if its state needs to change upon activation.
        cls._active_network_id = network_id
        if network_id in cls._cache:
            del cls._cache[network_id]

        print(f"Network {network_id} is now active. Cache updated.")


# --- Base Views ---

class BaseBrainAPIView(APIView):
    """Base view for handling service retrieval for a specific network."""
    permission_classes = [AllowAny]  # TODO: Change to IsAuthenticated in production

    def get_service(self, pk: int) -> Union[STAGNetworkService, Response]:
        """Provides a consistent way to get the service and handle errors."""
        try:
            return ServiceCache.get(pk)
        except BrainNetwork.DoesNotExist:
            return Response({"error": f"BrainNetwork with id {pk} not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            # Catching generic exceptions for robustness
            return Response({"error": f"An unexpected error occurred while loading the service: {str(e)}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --- ViewSets ---

class BrainNetworkViewSet(ModelViewSet):
    """API endpoint for managing BrainNetwork instances (CRUD)."""
    queryset = BrainNetwork.objects.all().order_by('-created_at')
    permission_classes = [AllowAny]  # TODO: Change to IsAdminUser in production

    def get_serializer_class(self):
        """Use a detailed serializer for retrieving a single network."""
        if self.action == 'retrieve':
            return BrainNetworkDetailSerializer
        return BrainNetworkSerializer

    @action(detail=True, methods=['post'], url_path='set-active')
    def set_active(self, request, pk=None):
        """Sets this network as the single active one."""
        network = self.get_object()
        ServiceCache.set_active(network.id)
        serializer = self.get_serializer(network)
        return Response({
            "message": f"Network {network.id} is now active.",
            "network": serializer.data
        }, status=status.HTTP_200_OK)

    @action(detail=True, methods=['get'], url_path='graph-state')
    def graph_state(self, request, pk=None):
        """Gets the current graph state for visualization."""
        service = ServiceCache.get(int(pk))
        graph_data = service.get_graph_state_for_viz()
        return Response(graph_data, status=status.HTTP_200_OK)


class SnapshotViewSet(ReadOnlyModelViewSet):
    """API endpoint for viewing and loading GraphSnapshots for a specific network."""
    serializer_class = GraphSnapshotSerializer
    permission_classes = [AllowAny]  # TODO: Change to IsAuthenticated

    def get_queryset(self):
        """Filters snapshots based on the network_id from the nested URL."""
        network_id = self.kwargs.get('network_pk')
        return GraphSnapshot.objects.filter(network_id=network_id).order_by('-created_at')

    @action(detail=True, methods=['post'], url_path='load')
    def load(self, request, network_pk=None, pk=None):
        """Loads a snapshot into its parent network's service instance."""
        service = ServiceCache.get(int(network_pk))
        try:
            service.load_from_snapshot(int(pk))
            return Response({"message": f"Successfully loaded snapshot {pk} into network {network_pk}."},
                            status=status.HTTP_200_OK)
        except GraphSnapshot.DoesNotExist:
            return Response({"error": "Snapshot not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": f"Failed to load snapshot: {str(e)}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TrainingSessionViewSet(ReadOnlyModelViewSet):
    """API endpoint for viewing the training history of a specific network."""
    serializer_class = TrainingSessionSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        """Filters sessions based on the network_id from the nested URL."""
        network_id = self.kwargs.get('network_pk')
        return TrainingSession.objects.filter(network_id=network_id).order_by('-start_time')


# --- Action-Oriented API Views ---

class StartTrainingView(BaseBrainAPIView):
    """API endpoint for starting a new training session on an environment."""

    def post(self, request, pk, *args, **kwargs):
        service = self.get_service(pk)
        if isinstance(service, Response): return service

        env_name = request.data.get('environment', 'gridworld')
        episodes = int(request.data.get('episodes', 50))
        max_steps = int(request.data.get('max_steps', 200))

        from .environments import EnvironmentManager
        try:
            env = EnvironmentManager.get_env(env_name)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        session = TrainingSession.objects.create(
            network_id=pk,
            environment_name=env_name,
            parameters={'episodes': episodes, 'max_steps': max_steps},
            status=TrainingSession.Status.RUNNING
        )

        try:
            # The train_on_env is a generator; we consume it to run the training.
            final_results = list(service.train_on_env(env, episodes, max_steps))

            session.status = TrainingSession.Status.COMPLETED
            session.end_time = timezone.now()
            session.results = {"episodes": final_results}
            session.save()

            return Response({
                "message": "Training completed successfully.",
                "session": TrainingSessionSerializer(session).data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            session.status = TrainingSession.Status.FAILED
            session.end_time = timezone.now()
            session.results = {"error": str(e)}
            session.save()
            return Response({
                "error": f"Training failed: {str(e)}",
                "session_id": session.id
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetEnvironmentsView(APIView):
    """API endpoint to get a list of available training environments."""
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        from .environments import EnvironmentManager
        descriptions = EnvironmentManager.get_environment_descriptions()
        return Response(descriptions, status=status.HTTP_200_OK)
