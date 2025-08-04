from django.urls import re_path
from . import consumers

# This URL pattern is now aligned with the one used in the STAGNetworkService
# and the frontend visualization components.
websocket_urlpatterns = [
    re_path(r'ws/brain/training/(?P<network_id>\w+)/$', consumers.GameTrainingConsumer.as_asgi()),
]
