"""Flask-Limiter instance setup for ResumeFit."""

import os

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

storage_uri = os.environ.get("RESUMEFIT_RATE_LIMIT_STORAGE_URL", "memory://")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60 per minute"],
    storage_uri=storage_uri,
)
