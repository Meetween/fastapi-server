import re

import pytest

from cli import create_admin, generate_random_password, main
from src.database.models.user import User


@pytest.mark.unit
class TestGenerateRandomPassword:
    def test_default_length(self):
        password = generate_random_password()
        assert len(password) == 20
        assert re.fullmatch(r"[A-Za-z0-9]+", password)


@pytest.mark.unit
class TestCreateAdminCli:
    @pytest.mark.asyncio
    async def test_create_admin_success(self, capsys):
        code = await create_admin(
            "cli-admin@example.com",
            password="secret123",
            first_name="CLI",
            last_name="Admin",
        )
        assert code == 0

        output = capsys.readouterr().out
        assert "Email: cli-admin@example.com" in output
        assert "Password: secret123" in output

        user = await User.find_by_email("cli-admin@example.com")
        assert user is not None
        assert user.is_admin
        assert user.first_name == "CLI"
        assert user.last_name == "Admin"

    @pytest.mark.asyncio
    async def test_create_admin_generates_password_when_omitted(self, capsys):
        code = await create_admin("generated-admin@example.com")
        assert code == 0

        output = capsys.readouterr().out
        assert "Email: generated-admin@example.com" in output
        match = re.search(r"Password: ([A-Za-z0-9]+)", output)
        assert match is not None
        assert len(match.group(1)) == 20

        user = await User.find_by_email("generated-admin@example.com")
        assert user is not None
        assert user.is_admin

    @pytest.mark.asyncio
    async def test_create_admin_duplicate_email(self, capsys):
        first = await create_admin(
            "dup-admin@example.com",
            password="secret123",
        )
        assert first == 0
        capsys.readouterr()

        second = await create_admin(
            "dup-admin@example.com",
            password="other-secret",
        )
        assert second == 1
        assert "Email dup-admin@example.com already exists" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_create_admin_rejects_empty_email(self, capsys):
        code = await create_admin("")
        assert code == 1
        assert "Email is required" in capsys.readouterr().err

    def test_main_dispatches_create_admin(self, capsys):
        """main() owns asyncio.run(); keep one sync test for argparse wiring."""
        code = main(
            [
                "create-admin",
                "main-entry@example.com",
                "--password",
                "secret123",
            ]
        )
        assert code == 0
        assert "Email: main-entry@example.com" in capsys.readouterr().out
