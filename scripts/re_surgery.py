import psycopg2
import pandas as pd

# Database Connection
db_config = {
    "dbname": "sgpgi_cdm",
    "user": "akshat",
    "password": "aksh4567", 
    "host": "localhost"
}

try:
    conn = psycopg2.connect(**db_config)
    
    # SQL Query: Find procedures performed > 1 time on the SAME patient
    query = """
    SELECT 
        c.concept_name AS procedure_name,
        p.procedure_concept_id,
        p.person_id,
        COUNT(p.procedure_occurrence_id) AS occurrence_count
    FROM cvts_cdm.procedure_occurrence p
    JOIN cvts_cdm.concept c 
        ON p.procedure_concept_id = c.concept_id
    GROUP BY p.person_id, c.concept_name, p.procedure_concept_id
    HAVING COUNT(p.procedure_occurrence_id) > 1
    ORDER BY occurrence_count DESC;
    """

    print("--- Searching for Repeat Procedures ---")
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No repeat procedures found. (Did you import 'procedure_occurrence.csv' yet?)")
    else:
        print(df)
    
    df.to_csv("data/out/re_surgery.csv")


except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()