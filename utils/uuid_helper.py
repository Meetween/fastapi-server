import uuid

class UUIDHelper:

    @classmethod
    def uuid_from_int(cls, n: int) -> uuid.UUID:
        return uuid.uuid5(uuid.NAMESPACE_OID, str(n))
