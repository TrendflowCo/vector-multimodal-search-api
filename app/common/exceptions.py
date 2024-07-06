class Error(Exception):
    """Base class for other exceptions"""
    pass

class DataNotFoundError(Error):
    """Raised when the data is not found in the database or storage"""
    def __init__(self, message="Data not found"):
        self.message = message
        super().__init__(self.message)

class InvalidUsageError(Error):
    """Raised when the usage of a function or operation is invalid"""
    def __init__(self, message="Invalid usage"):
        self.message = message
        super().__init__(self.message)

class ServiceUnavailableError(Error):
    """Raised when an external service is unavailable"""
    def __init__(self, message="Service is currently unavailable"):
        self.message = message
        super().__init__(self.message)

class InvalidParameterError(Error):
    """Raised when the given parameters are invalid"""
    def __init__(self, message="Invalid parameters"):
        self.message = message
        super().__init__(self.message)

class AuthenticationError(Error):
    """Raised when there is a failure in authentication"""
    def __init__(self, message="Authentication failed"):
        self.message = message
        super().__init__(self.message)

class AuthorizationError(Error):
    """Raised when there is a failure in authorization"""
    def __init__(self, message="Authorization failed"):
        self.message = message
        super().__init__(self.message)

class ExternalAPIError(Error):
    """Raised when an external API fails to execute properly"""
    def __init__(self, message="External API error"):
        self.message = message
        super().__init__(self.message)
