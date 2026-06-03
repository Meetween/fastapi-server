import pytest

from src.database.models.user import User
from utils.auth import create_user


@pytest.mark.unit
class TestCreateUser:
    @pytest.mark.asyncio
    async def test_creates_regular_user(self):
        user = await create_user("user@example.com", "secret123")

        assert user.email == "user@example.com"
        assert user.role == "user"
        assert not user.is_admin

    @pytest.mark.asyncio
    async def test_creates_admin_user(self):
        user = await create_user("admin@example.com", "secret123", is_admin=True)

        assert user.email == "admin@example.com"
        assert user.role == "admin"
        assert user.is_admin

    @pytest.mark.asyncio
    async def test_normalizes_email(self):
        user = await create_user("MixedCase@Example.com", "secret123")

        assert user.email == "mixedcase@example.com"
        stored = await User.find_by_email("mixedcase@example.com")
        assert stored is not None

    @pytest.mark.asyncio
    async def test_rejects_duplicate_email(self):
        await create_user("dup@example.com", "secret123")

        with pytest.raises(ValueError, match="already exists"):
            await create_user("dup@example.com", "other-secret")
