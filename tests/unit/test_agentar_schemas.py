"""Request model rules that the contract states and Pydantic can enforce."""
import pytest
from pydantic import ValidationError

from src.schemas.agentar import (
    AgentarWorkerBot,
    MeetingCreate,
    MeetingPatch,
    MePatch,
    WorkerEvent,
)
from src.schemas.auth import LoginRequest, SignupRequest

ID = "0123456789abcdef01234567"


class TestMePatch:
    def test_empty_body_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            MePatch()
        with pytest.raises(ValidationError):
            MePatch(profile={})

    def test_explicit_null_is_distinct_from_omitted(self) -> None:
        patch = MePatch(profile={"agent_name": None})
        assert patch.profile.model_fields_set == {"agent_name"}
        assert patch.profile.agent_name is None

    @pytest.mark.parametrize("name", ["Niccol\u00f2", "Zo\u00eb 2", "O'Brien", "Anne-Marie", "\u7530\u4e2d", "a" * 32])
    def test_agent_name_accepts_unicode_letters(self, name: str) -> None:
        assert MePatch(profile={"agent_name": name}).profile.agent_name == name

    @pytest.mark.parametrize(
        "name",
        ["Bad (AI)", "", " lead", "x" * 33, "Agentar", "my AGENTAR bot", "a\u200bb", "a\nb", "a/b", "-dash", "a<b>"],
    )
    def test_agent_name_rule(self, name: str) -> None:
        with pytest.raises(ValidationError):
            MePatch(profile={"agent_name": name})

    @pytest.mark.parametrize("name", ["Laura (AI)", "Agentar", "x\u202eevil", "a\x00b", ""])
    def test_person_names_cannot_fake_or_hide_the_disclosure(self, name: str) -> None:
        with pytest.raises(ValidationError):
            MePatch(first_name=name)
        with pytest.raises(ValidationError):
            MePatch(last_name=name)

    def test_person_names_keep_ordinary_punctuation(self) -> None:
        assert MePatch(last_name="O\u2019Neill-Rossi Jr.").last_name == "O\u2019Neill-Rossi Jr."

    def test_unknown_voice_and_extra_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            MePatch(profile={"voice": "bf_emma"})
        with pytest.raises(ValidationError):
            MePatch(is_admin=True)

    def test_complete_onboarding_only_true(self) -> None:
        assert MePatch(complete_onboarding=True).complete_onboarding is True
        with pytest.raises(ValidationError):
            MePatch(complete_onboarding=False)


class TestMeetings:
    def test_create_needs_room_and_mode(self) -> None:
        with pytest.raises(ValidationError):
            MeetingCreate(room="weekly")
        created = MeetingCreate(room="weekly", mode="now")
        assert created.grace_minutes == 15
        assert created.questions == []

    def test_patch_tracks_only_the_sent_fields(self) -> None:
        patch = MeetingPatch(title="t")
        assert patch.model_dump(exclude_unset=True) == {"title": "t"}
        with pytest.raises(ValidationError):
            MeetingPatch()

    @pytest.mark.parametrize(
        "field",
        [
            {"grace_minutes": 0},
            {"grace_minutes": 61},
            {"questions": ["q"] * 11},
            {"questions": ["x" * 301]},
            {"questions": [""]},
            {"purpose": "x" * 1001},
            {"relevant_info": "x" * 2001},
            {"file_ids": [ID, ID]},
            {"file_ids": ["not-an-id"]},
            {"unknown": 1},
        ],
    )
    def test_field_limits(self, field: dict) -> None:
        with pytest.raises(ValidationError):
            MeetingCreate(room="weekly", mode="now", **field)


class TestSignup:
    BODY = {"email": "Laura@Example.org", "password": "correct horse", "first_name": "Laura", "last_name": "Rossi"}

    def test_email_is_lower_cased(self) -> None:
        assert SignupRequest(**self.BODY).email == "laura@example.org"

    @pytest.mark.parametrize("email", ["nobody", "a@b", "@x.org", "a b@x.org", "a@@x.org"])
    def test_bad_email(self, email: str) -> None:
        with pytest.raises(ValidationError):
            SignupRequest(**{**self.BODY, "email": email})

    @pytest.mark.parametrize("name", ["Laura (AI)", "agentar", "Ann\u200b"])
    def test_names_cannot_fake_the_disclosure(self, name: str) -> None:
        with pytest.raises(ValidationError):
            SignupRequest(**{**self.BODY, "first_name": name})
        with pytest.raises(ValidationError):
            SignupRequest(**{**self.BODY, "last_name": name})

    def test_password_length(self) -> None:
        with pytest.raises(ValidationError):
            SignupRequest(**{**self.BODY, "password": "short"})


class TestWorkerEvent:
    BASE = {
        "event_id": "6f1c2f1e-0f3a-4c52-9a43-0a6f4e3d9b11",
        "agentar_meeting_id": ID,
        "bot_id": "bot1",
        "at": "2026-11-10T09:00:00Z",
    }

    def test_data_is_chosen_by_kind(self) -> None:
        event = WorkerEvent(**self.BASE, type="escalation", data={"kind": "escalation", "question": "q", "asker": None})
        assert event.data.kind == "escalation"

    def test_type_must_match_kind(self) -> None:
        with pytest.raises(ValidationError):
            WorkerEvent(**self.BASE, type="left", data={"kind": "joined"})

    def test_extra_data_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            WorkerEvent(**self.BASE, type="joined", data={"kind": "joined", "token": "secret"})

    def test_bad_bot_id(self) -> None:
        with pytest.raises(ValidationError):
            WorkerEvent(**{**self.BASE, "bot_id": "../x"}, type="joined", data={"kind": "joined"})


class TestWorkerBot:
    def test_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            AgentarWorkerBot(room="weekly", surprise=1)

    def test_script_and_opener_are_exclusive(self) -> None:
        with pytest.raises(ValidationError):
            AgentarWorkerBot(room="weekly", script="a", opener="b")

    def test_meeting_id_needs_events_url(self) -> None:
        with pytest.raises(ValidationError):
            AgentarWorkerBot(room="weekly", agentar_meeting_id=ID)
        AgentarWorkerBot(room="weekly", agentar_meeting_id=ID, events_url="http://f:8000/api/agentar/events")


class TestLoginRequest:
    def test_length_caps(self) -> None:
        LoginRequest(email="a@b.co", password="x" * 128)
        for body in (
            {"email": "a" * 255, "password": "x"},
            {"email": "a@b.co", "password": "x" * 129},
            {"email": "", "password": "x"},
            {"email": "a@b.co", "password": ""},
        ):
            with pytest.raises(ValidationError):
                LoginRequest(**body)


class TestWorkerDisplayName:
    @pytest.mark.parametrize("name", ["Laura Agentar", "Niccol\u00f2 Agentar", "Agentar", "O'Brien Agentar"])
    def test_accepts_names_with_the_agentar_word(self, name: str) -> None:
        assert AgentarWorkerBot(room="weekly", display_name=name).display_name == name

    @pytest.mark.parametrize("name", ["Laura (AI)", "", "x" * 49, "a\u200bb", "-x", "a/b"])
    def test_refuses_parentheses_and_odd_characters(self, name: str) -> None:
        with pytest.raises(ValidationError):
            AgentarWorkerBot(room="weekly", display_name=name)
