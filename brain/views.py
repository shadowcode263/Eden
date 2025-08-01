from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import BrainNetwork, BrainPattern, BrainRetrieval
from .serializers import (
    BrainNetworkSerializer, BrainPatternSerializer, BrainRetrievalSerializer,
    StorePatternSerializer, RetrievePatternSerializer
)
from .services import HopfieldNetworkService


class BrainNetworkViewSet(viewsets.ModelViewSet):
    queryset = BrainNetwork.objects.all()
    serializer_class = BrainNetworkSerializer

    def partial_update(self, request, pk=None):
        """Update network parameters"""
        network = self.get_object()
        serializer = self.get_serializer(network, data=request.data, partial=True)

        if serializer.is_valid():
            # Clear cache when parameters change
            from django.core.cache import cache
            cache_key = f"network_{network.id}_state"
            cache.delete(cache_key)

            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def store_pattern(self, request, pk=None):
        """Store a new pattern in the network"""
        network = self.get_object()
        serializer = StorePatternSerializer(data=request.data)

        if serializer.is_valid():
            service = HopfieldNetworkService(network.id)
            result = service.store_pattern(serializer.validated_data['text'])

            if result['success']:
                return Response(result, status=status.HTTP_201_CREATED)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def retrieve_pattern(self, request, pk=None):
        """Retrieve a pattern from the network"""
        network = self.get_object()
        serializer = RetrievePatternSerializer(data=request.data)

        if serializer.is_valid():
            service = HopfieldNetworkService(network.id)
            result = service.retrieve_pattern(
                serializer.validated_data['query_text'],
                serializer.validated_data['max_iter']
            )

            return Response(result, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        """Get network statistics"""
        network = self.get_object()
        service = HopfieldNetworkService(network.id)
        stats = service.get_network_stats()

        return Response(stats, status=status.HTTP_200_OK)

    @action(detail=True, methods=['get'])
    def patterns(self, request, pk=None):
        """Get all patterns for a network"""
        network = self.get_object()
        patterns = network.patterns.all().order_by('-created_at')
        serializer = BrainPatternSerializer(patterns, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=True, methods=['get'])
    def retrievals(self, request, pk=None):
        """Get all retrievals for a network"""
        network = self.get_object()
        retrievals = network.retrievals.all().order_by('-created_at')[:50]  # Last 50
        serializer = BrainRetrievalSerializer(retrievals, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], url_path='last-retrieval-firing-data/(?P<id>[^/.]+)')
    def get_last_retrieval_firing_data(self, request, id=None):
        """
        Get the neuron firing data (retrieval steps) for the most recent retrieval
        of a specific network.
        """
        network = get_object_or_404(BrainNetwork, id=id)
        latest_retrieval = BrainRetrieval.objects.filter(network=network).order_by('-created_at').first()

        if not latest_retrieval:
            return Response({'message': 'No retrieval data found for this network.'}, status=status.HTTP_404_NOT_FOUND)

        serializer = BrainRetrievalSerializer(latest_retrieval)
        return Response(serializer.data, status=status.HTTP_200_OK)
