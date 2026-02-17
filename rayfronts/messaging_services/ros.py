import threading
from functools import partial
import logging

import std_msgs.msg
from rayfronts.messaging_services import MessagingService

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from rclpy.qos import QoSProfile, ReliabilityPolicy
import std_msgs

logger = logging.getLogger(__name__)

class Ros2MessagingService(MessagingService):

  def __init__(self,
               text_query_topic,
               text_query_callback = None,
               sync_query_callback = None,
               gps_topic = None,
               **kwargs):
    super().__init__()
    self.text_query_topic = text_query_topic
    self.text_query_callback = text_query_callback
    self.gps_topic = gps_topic

    if not rclpy.ok():
      rclpy.init()
    self._rosnode = Node("rayfronts_messaging_service")

    self.text_query_sub = self._rosnode.create_subscription(
      std_msgs.msg.String, text_query_topic, self.text_query_handler,
      QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=5))

    self._ros_executor = SingleThreadedExecutor()
    self._ros_executor.add_node(self._rosnode)
    self._spin_thread = threading.Thread(
      target=self._spin_ros, name="rayfronts_messaging_service_spinner")
    self._spin_thread.daemon = True
    self._spin_thread.start()

    logger.info("Messaging Service initialized successfully.")

  def _spin_ros(self):
    try:
      self._ros_executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass

  def text_query_handler(self, s):
    if self.text_query_callback is not None:
      self.text_query_callback(s.data)

  def broadcast_gps_message(self, lat, long):
    raise NotImplementedError()

  def broadcast_map_update(self, map_update):
    raise NotImplementedError()

  def join(self, timeout = None):
    self._spin_thread.join(timeout)

  def shutdown(self):
    self._rosnode.context.try_shutdown()
