import logging
logger = logging.getLogger(__name__)

from rayfronts.datasets.base import PosedRgbdDataset, SemSegDataset
from rayfronts.datasets.replica import NiceReplicaDataset, SemanticNerfReplicaDataset
from rayfronts.datasets.ros import RosnpyDataset, Ros2Subscriber, StarlingMaxSubscriber
from rayfronts.datasets.scannet import ScanNetDataset
from rayfronts.datasets.tartanair import TartanAirDataset

failed_to_import = list()
try:
  from rayfronts.datasets.dummy import DummyDataset
except:
  failed_to_import.append("DummyDataset")

try:
  from rayfronts.datasets.airsim import AirSimDataset
except:
  failed_to_import.append("AirSimDataset")

if len(failed_to_import) > 0:
  logger.info(
    "Could not import %s. Make sure you have their extra dependencies "
    "installed if you want to use them.", ", ".join(failed_to_import))
