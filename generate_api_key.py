import secrets
import string

def generate_api_key(length=32):
    """Generate a secure random API key."""
    alphabet = string.ascii_letters + string.digits
    api_key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return api_key

if __name__ == "__main__":
    api_key = generate_api_key()
    print(f"Generated API Key: {api_key}")
    print("Make sure to add this key to your .env file as API_KEY=<generated_key>")
