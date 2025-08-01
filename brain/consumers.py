import json
from channels.generic.websocket import AsyncWebsocketConsumer
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class NetworkConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.network_id = self.scope['url_route']['kwargs']['network_id']
        self.network_group_name = f'network_{self.network_id}'

        logger.info(f"WebSocket connecting for network_id: {self.network_id}")

        # Join network group
        await self.channel_layer.group_add(
            self.network_group_name,
            self.channel_name
        )

        await self.accept()
        logger.info(f"WebSocket connected for network_id: {self.network_id}")

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnecting for network_id: {self.network_id} with code: {close_code}")
        # Leave network group
        await self.channel_layer.group_discard(
            self.network_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')
            logger.debug(f"Received message type: {message_type} for network_id: {self.network_id}")

            if message_type == 'ping':
                # Respond to ping with pong
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': text_data_json.get('timestamp', datetime.now().isoformat())
                }))
            elif message_type == 'heartbeat':
                # Handle heartbeat from client
                await self.send(text_data=json.dumps({
                    'type': 'heartbeat_ack',
                    'timestamp': text_data_json.get('timestamp', datetime.now().isoformat())
                }))
            # Add other message types if needed for client-to-server communication

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def network_update(self, event):
        # Send message to WebSocket
        logger.debug(f"Sending network_update event: {event['event_type']} to network_id: {self.network_id}")
        try:
            await self.send(text_data=json.dumps({
                'type': 'network_update',
                'event_type': event['event_type'],
                'data': event['data'],
                'timestamp': datetime.now().isoformat()
            }))
        except Exception as e:
            logger.error(f"Error sending network update: {e}")
