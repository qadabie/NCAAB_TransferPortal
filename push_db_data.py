import pandas as pd
from sqlalchemy import create_engine
from db_connect import get_connection

def get_engine():
    conn = get_connection()
    return create_engine(f'postgresql+psycopg2://{conn.info.user}:{conn.info.password}@{conn.info.host}:{conn.info.port}/{conn.info.dbname}')

def push_data_to_db(df):
    engine = get_engine()

    table_name = input("\nEnter the name of the table to upload to: ").strip()

    # Prompt user for action
    action = input(f"\nDo you want to [a]ppend or [r]eplace the table '{table_name}'? (a/r): ").strip().lower()
    if action == 'a':
        mode = 'append'
    elif action == 'r':
        mode = 'replace'
    else:
        print("Invalid input. Defaulting to 'replace'.")
        mode = 'replace'

    # Push data
    df.to_sql(table_name, engine, if_exists = mode, index = False)
    print(f"Data {mode}d to table '{table_name}'.")
