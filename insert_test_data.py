import os
from dotenv import load_dotenv
import psycopg2
# Load .env file
load_dotenv()

# Connect using environment variables
conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    dbname=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    sslmode=os.getenv("PG_SSLMODE")
)

cursor = conn.cursor()

# Insert test data
test_data = [
    ("Quinen", 25),
    ("Jacob", 34),
    ("Alex", 25),
    ("Nial", 25)
]

for name, age in test_data:
    cursor.execute(
        "INSERT INTO test (name, age) VALUES (%s, %s);",
        (name, age)
    )

conn.commit()
cursor.close()
conn.close()

print("Test data inserted.")