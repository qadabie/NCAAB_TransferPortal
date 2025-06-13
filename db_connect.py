import os
from dotenv import load_dotenv
import psycopg2

# Load .env file
load_dotenv()

def get_connection():
    return psycopg2.connect(
        host = os.getenv("PG_HOST"),
        port = os.getenv("PG_PORT"),
        dbname = os.getenv("PG_DATABASE"),
        user = os.getenv("PG_USER"),
        password = os.getenv("PG_PASSWORD"),
        sslmode = os.getenv("PG_SSLMODE")
    )