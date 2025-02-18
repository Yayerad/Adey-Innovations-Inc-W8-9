import logging
from flask.logging import default_handler

def configure_logging(app):
    """Configure logging for Flask application"""
    # Remove default handler
    app.logger.removeHandler(default_handler)
    
    # File handler
    file_handler = logging.FileHandler('fraud_api.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Add handlers
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.DEBUG)