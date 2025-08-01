from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BrainNetworkViewSet

router = DefaultRouter()
router.register(r'networks', BrainNetworkViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
