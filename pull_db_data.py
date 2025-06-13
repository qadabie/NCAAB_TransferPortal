import pandas as pd
from db_connect import get_connection

def fetch_data(table_name):
    conn = get_connection()
    query = f"SELECT * FROM {table_name};"
    try:
        df = pd.read_sql(query, conn)
        print(f"Pulled {len(df)} rows from table '{table_name}'")
    except Exception as e:
        print(f"Error pulling data: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

if __name__ == "__main__":
    table = input("Enter the name of the table to fetch data from: ").strip()
    df = fetch_data(table)
    print(df.head())