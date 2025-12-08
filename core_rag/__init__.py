__version__ = "0.1.0"

from . import ingestion
from . import retrieval
from . import utils

try:
    from . import summary
    SUMMARY_AVAILABLE = True
except ImportError:
    summary = None
    SUMMARY_AVAILABLE = False

__all__ = ['ingestion', 'retrieval', 'utils', 'summary', '__version__', 'SUMMARY_AVAILABLE']
