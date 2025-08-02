from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/network/(?P<network_id>\w+)/$', consumers.BrainNetworkConsumer.as_asgi()),
]
