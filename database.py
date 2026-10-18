import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:supersecret0404@localhost/amber_ai_database")

# Ensure postgresql:// prefix (some providers like Render output postgres://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
