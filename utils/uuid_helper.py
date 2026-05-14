from uuid import uuid4, UUID, uuid5, NAMESPACE_OID


def new_meeting_id() -> str:
    """Return a new RFC 4122 UUID string for meeting identifiers (same shape as test `generate_test_meeting_id`)."""
    return str(uuid4())


class UUIDHelper:

    @classmethod
    def uuid_from_int(cls, n: int) -> UUID:
        return uuid5(NAMESPACE_OID, str(n))
