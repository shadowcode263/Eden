import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer


class GameTrainingConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer to handle real-time game training sessions and visualization.
    This consumer acts as a simple relay, forwarding broadcasted events from the
    STAGNetworkService to the connected client.
    """

    async def connect(self):
        """Handles new WebSocket connections."""
        self.network_id = self.scope['url_route']['kwargs']['network_id']
        self.room_group_name = f'brain_training_{self.network_id}'

        # Add the consumer to the group for this network
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()
        print(f"WebSocket connection established for network {self.network_id} in room {self.room_group_name}")

    async def disconnect(self, close_code):
        """Handles WebSocket disconnections."""
        # Remove the consumer from the group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        print(f"WebSocket connection closed for room {self.room_group_name}")

    async def receive(self, text_data):
        """
        Handles incoming messages. For this simple relay, we don't need to
        process any incoming messages, but the method is here for future use.
        """
        pass

    async def broadcast_event(self, event: dict):
        """
        Handler for messages sent to this consumer's group from the backend service.
        It forwards the payload directly to the client's WebSocket.
        """
        payload = event['payload']
        # The payload contains 'event_type' and the data for the frontend
        await self.send(text_data=json.dumps(payload))
