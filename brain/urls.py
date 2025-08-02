from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    BrainNetworkViewSet,
    LearnTextView,
    LearnBookView,  # Import the new view
    PredictStoryContinuationView,
    GraphStateView,
    SnapshotViewSet,
    AdminActionsView,
    GameTrainingView,  # Import the new view
    list_networks,  # Add the new view function
    create_network,  # Add the new view function
    network_action,  # Add the new view function
)

app_name = 'brain'

# The router generates standard URLs for the ViewSets.
# The frontend will call '/api/brain/networks/' and '/api/brain/snapshots/'.
router = DefaultRouter()
router.register(r'networks', BrainNetworkViewSet, basename='brain-network')
router.register(r'snapshots', SnapshotViewSet, basename='graph-snapshot')

urlpatterns = [
    # Router-generated URLs are included first.
    path('', include(router.urls)),

    # Custom network URLs
    path('networks/', list_networks, name='list_networks'),
    path('networks/create/', create_network, name='create_network'),
    path('networks/<int:network_id>/actions/', network_action, name='network_action'),
    path('networks/<int:network_id>/set-active/', network_action, name='set_active_network'),  # Add this line

    # Core learning and inference endpoints.
    path('actions/learn/', LearnTextView.as_view(), name='learn-text'),
    path('actions/learn-book/', LearnBookView.as_view(), name='learn-book'),  # Add the new URL
    path('actions/predict/', PredictStoryContinuationView.as_view(), name='predict-story'),
    path('actions/game-training/', GameTrainingView.as_view(), name='game-training'),  # Use the class-based view

    # Visualization endpoint.
    path('state/', GraphStateView.as_view(), name='graph-state'),

    # Admin utility endpoint.
    path('admin/actions/', AdminActionsView.as_view(), name='admin-actions'),
]
