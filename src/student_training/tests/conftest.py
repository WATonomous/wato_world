import pytest


@pytest.fixture
def fixtures_dir(request):
    return request.path.parent / "fixtures"
