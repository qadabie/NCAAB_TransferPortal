from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load .env file
load_dotenv()

# as of 7/22/25 the database information is hardcoded so the instructional team can run the code
# Advanced uses of this method would involve a personal .env file for each user using the following function

# def get_connection():
#     return create_engine(
#         f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
#         f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
#     )

def get_connection():
    return create_engine(
        "postgresql+psycopg2://avnadmin:AVNS_qm_fxCQpBhTyFqQeDEq"
        "@umich-siads699-ncaab-transferportal-umich-siads699-ncaab-transf.f.aivencloud.com:14357/defaultdb"
    )