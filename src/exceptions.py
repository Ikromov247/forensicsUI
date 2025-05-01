class ApplicationError(Exception):
    """Base exception for the application"""
    pass

class ForensicsError(ApplicationError):
    """Base exception for the application"""
    pass

class DetectionError(ForensicsError):
    """Raised when object detection fails"""
    pass

class FeatureExtractionError(ForensicsError):
    """Raised when feature extraction fails"""
    pass

class DatabaseError(ForensicsError):
    """Raised when database operations fail"""
    pass

class ConfigurationError(ForensicsError):
    """Raised when configuration is invalid"""
    pass

class InputError(ForensicsError):
    """Raised when input data is invalid"""
    pass

class VisualizationError(ForensicsError):
    """Raised when visualization fails"""
    pass

class ResourceError(ForensicsError):
    """Raised when resource allocation fails"""
    pass 