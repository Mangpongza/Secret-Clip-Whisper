class WhisperStegError(Exception):
    """ข้อผิดพลาดหลักของแอปที่สามารถแสดงแก่ผู้ใช้ได้"""


class DependencyError(WhisperStegError):
    pass


class CapacityError(WhisperStegError):
    pass


class AuthenticationError(WhisperStegError):
    pass


class PayloadFormatError(WhisperStegError):
    pass
