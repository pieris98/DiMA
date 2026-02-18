from src.encoders.base import Encoder
from src.encoders.esm2 import ESM2EncoderModel
from src.encoders.saprot import SaprotEncoderModel
from src.encoders.enc_normalizer import EncNormalizer

try:
    from src.encoders.cheap import CHEAPEncoderModel
except ImportError:
    pass
