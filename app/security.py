from flask_httpauth import HTTPTokenAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from app.config import API_KEY

auth = HTTPTokenAuth(scheme='Bearer')

@auth.verify_token
def verify_token(token):
    if token == API_KEY:
        return True
    return False

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000000 per day", "1000000 per hour"]
)
