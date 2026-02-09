from src.encoders.base import Encoder
from src.encoders.esm2 import ESM2EncoderModel
from src.encoders.saprot import SaprotEncoderModel
# from src.encoders.esmc import ESMCEncoderModel
from src.encoders.cheap import CHEAPEncoderModel
from src.encoders.enc_normalizer import EncNormalizer

# Register built-in encoders with the pipeline registry
from pipeline.registry import registry

registry.register("encoder", "esm2", ESM2EncoderModel)
registry.register("encoder", "saprot", SaprotEncoderModel)
registry.register("encoder", "cheap", CHEAPEncoderModel)
