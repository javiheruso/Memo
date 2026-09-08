"""Re-export of the shared example-data seed universe.

The seed data now lives in :mod:`app.services.seed` so the production app can
reuse it for the admin "load example data" feature. This module keeps the
historical import path working so existing tests stay unchanged.
"""

from app.services.seed import SeedUniverse, seed_universe

__all__ = ["SeedUniverse", "seed_universe"]
