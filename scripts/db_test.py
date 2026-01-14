import psycopg2
import pandas as pd

def search_concept(keyword):
    """
    Searches the OMOP Concept table for a specific medical term.
    """
    # 1. Connect (Using Unix Socket for speed)
    conn = psycopg2.connect(
        dbname="sgpgi_cdm",
        user="akshat",
        password="aksh4567", # Add password if you set one
        host="localhost"
    )

    # 2. SQL Query: specific to OMOP structure
    # We look for Standard concepts (S) that match your keyword
    query = f"""
    SELECT concept_id, concept_name, domain_id, vocabulary_id, concept_code
    FROM cvts_cdm.concept
    WHERE concept_name ILIKE '%{keyword}%' 
    AND standard_concept = 'S'
    LIMIT 10;
    """
    
    try:
        # 3. Load into Pandas for clean formatting
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print(f"No concepts found for '{keyword}'")
        else:
            print(f"\n--- Search Results for '{keyword}' ---")
            print(df[['concept_id', 'concept_name', 'domain_id', 'vocabulary_id']])
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Test 1: Search for a heart condition
    search_concept("Heart Failure")
    
    # Test 2: Search for a procedure
    search_concept("Bypass")