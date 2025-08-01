from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Ensure the network_id is captured as a digit
    re_path(r'ws/network/(?P<network_id>\d+)/$', consumers.NetworkConsumer.as_asgi()),
]
