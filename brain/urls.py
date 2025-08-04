from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    BrainNetworkViewSet,
    SnapshotViewSet,
    TrainingSessionViewSet,
    StartTrainingView,
    GetEnvironmentsView,
)

app_name = 'brain'

# --- Top-Level Router ---
# This handles the primary /networks/ endpoint.
router = DefaultRouter()
router.register(r'networks', BrainNetworkViewSet, basename='network')

# --- Manually Defined URL Patterns for Nested Resources ---
# We define these paths explicitly to avoid the rest_framework_nested dependency.
# This structure ensures the views still receive the 'network_pk' kwarg.
urlpatterns = [
    # Include the router-generated URLs for /networks/ and /networks/{pk}/
    path('', include(router.urls)),

    # --- Snapshots (nested under a specific network) ---
    path('networks/<int:network_pk>/snapshots/',
         SnapshotViewSet.as_view({'get': 'list'}),
         name='network-snapshot-list'),

    path('networks/<int:network_pk>/snapshots/<int:pk>/',
         SnapshotViewSet.as_view({'get': 'retrieve'}),
         name='network-snapshot-detail'),

    path('networks/<int:network_pk>/snapshots/<int:pk>/load/',
         SnapshotViewSet.as_view({'post': 'load'}),
         name='network-snapshot-load'),

    # --- Training Sessions (nested under a specific network) ---
    path('networks/<int:network_pk>/training-sessions/',
         TrainingSessionViewSet.as_view({'get': 'list'}),
         name='network-training-session-list'),

    path('networks/<int:network_pk>/training-sessions/<int:pk>/',
         TrainingSessionViewSet.as_view({'get': 'retrieve'}),
         name='network-training-session-detail'),

    # --- Custom Action-Oriented Endpoints ---

    # Endpoint to start a training session for a specific network
    path('networks/<int:pk>/train/', StartTrainingView.as_view(), name='network-train'),

    # Standalone endpoint to get available environments
    path('environments/', GetEnvironmentsView.as_view(), name='get-environments'),
]
