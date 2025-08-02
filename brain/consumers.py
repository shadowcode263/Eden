import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

class BrainNetworkConsumer(AsyncWebsocketConsumer):
  """
  Handles WebSocket connections for real-time updates of the STAG network.
  """
  async def connect(self):
      """
      Called when the websocket is trying to connect.
      """
      self.network_id = self.scope['url_route']['kwargs']['network_id']
      self.network_group_name = f'brain_{self.network_id}'

      logger.info(f"WebSocket connecting to group: {self.network_group_name}")

      await self.channel_layer.group_add(
          self.network_group_name,
          self.channel_name
      )

      await self.accept()
      logger.info(f"WebSocket connected for network_id: {self.network_id}")

  async def disconnect(self, close_code):
      """
      Called when the WebSocket closes for any reason.
      """
      logger.info(f"WebSocket disconnecting from group: {self.network_group_name}")
      await self.channel_layer.group_discard(
          self.network_group_name,
          self.channel_name
      )

  async def receive(self, text_data):
      """
      Called when a message is received from the WebSocket.
      This is primarily for health checks like ping/pong.
      """
      try:
          data = json.loads(text_data)
          message_type = data.get('type')

          if message_type == 'ping':
              await self.send(text_data=json.dumps({'type': 'pong'}))
      except json.JSONDecodeError:
          logger.warning(f"Received invalid JSON from client: {text_data}")
      except Exception as e:
          logger.error(f"Error in receive method: {e}")

  async def graph_event_batch(self, event):
      """
      This handler is kept for any legacy or specific batched events you might reintroduce.
      """
      logger.debug(f"Sending event batch to network_id: {self.network_id}")
      try:
          await self.send(text_data=json.dumps({
              'type': 'graph_event_batch',
              'events': event['events']
          }))
      except Exception as e:
          logger.error(f"Error sending graph event batch: {e}")

  async def graph_state_update(self, event):
      """
      FIX: This new method handles the 'graph_state_update' message type.
      It receives the full, updated graph state from the service and sends it
      to the frontend.
      """
      logger.debug(f"Sending graph_state_update to network_id: {self.network_id}")
      try:
          payload = event['payload']
          stats = {
              'total_nodes': len(payload['nodes']),
              'total_edges': len(payload['links']),
          }
          await self.send(text_data=json.dumps({
              'type': 'graph_state_update',
              'payload': payload,
              'stats': stats, # Add stats to the payload
          }))
      except Exception as e:
          logger.error(f"Error sending graph state update: {e}")
