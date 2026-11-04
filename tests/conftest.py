import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True, scope="session")
def _single_threaded_numerics():
    """Small model fits are fastest single-threaded; on a busy machine, OpenMP spin-waiting makes them stall."""
    with threadpool_limits(limits=1):
        yield
