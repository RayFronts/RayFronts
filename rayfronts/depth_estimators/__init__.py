from rayfronts.depth_estimators.base import DepthEstimator

import logging

logger = logging.getLogger(__name__)

from rayfronts.depth_estimators.lingbot_depth import LingbotDepthEstimator

failed_to_import = list()

if failed_to_import:
  logger.info(
      "Could not import %s. Make sure you have their submodules "
      "initialized and their extra dependencies installed if you "
      "want to use them.",
      ", ".join(failed_to_import),
  )

