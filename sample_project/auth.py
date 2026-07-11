def verify_password(plain: str, hashed: str) -> bool:
    """Compare a plaintext password against a bcrypt hash."""
    import bcrypt
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: str) -> str:
    """Issue a short-lived JWT for the given user."""
    import jwt
    return jwt.encode({"sub": user_id}, "secret", algorithm="HS256")