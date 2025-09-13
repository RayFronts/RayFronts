from rayfronts.image_encoders.base import (
  ImageEncoder, ImageGlobalEncoder,
  ImageSpatialEncoder, ImageSpatialGlobalEncoder,
  LangImageEncoder, LangGlobalImageEncoder,
  LangSpatialImageEncoder, LangSpatialGlobalImageEncoder,
  ImageSemSegEncoder
)

import logging
logger = logging.getLogger(__name__)

from rayfronts.image_encoders.radio import RadioEncoder
from rayfronts.image_encoders.naclip import NACLIPEncoder
from rayfronts.image_encoders.naradio import NARadioEncoder
from rayfronts.image_encoders.gt import GTEncoder
from rayfronts.image_encoders.semseg_wrap import SemSegWrapEncoder
failed_to_import = list()
try:
  from rayfronts.image_encoders.trident import TridentEncoder
except ModuleNotFoundError as e:
  failed_to_import.append("TridentEncoder")
try:
  from rayfronts.image_encoders.conceptfusion import ConceptFusionEncoder
except ModuleNotFoundError as e:
  failed_to_import.append("ConceptFusionEncoder")

try:
  from rayfronts.image_encoders.grounded_sam import GroundedSamSemSegEncoder
except ModuleNotFoundError as e:
  failed_to_import.append("GroundedSamSemSegEncoder")

if len(failed_to_import) > 0:
  logger.info("Could not import %s."
              "Make sure you have their submodules initialized and their"
              "extra dependencies installed if you want to use them.",
              ", ".join(failed_to_import))
