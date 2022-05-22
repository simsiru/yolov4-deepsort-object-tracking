from utils import DBInterface
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup database')

    parser.add_argument('--db_hostname', type=str, required=True,
    help='Postgresql database hostname')

    args = parser.parse_args()

    db_interface = DBInterface(password='docker', username='postgres', hostname=args.db_hostname, database='postgres', port_id=5432)

    sql_script = """
    CREATE TABLE face_embeddings_and_depth_maps (
        id SERIAL PRIMARY KEY,
        person_name VARCHAR(32) NOT NULL,
        face_embedding BYTEA NOT NULL,
        face_depth_map BYTEA
    )
    """
    db_interface.execute_sql_script(sql_script)
    sql_script = """
    SELECT *
    FROM face_embeddings_and_depth_maps
    """
    print(db_interface.execute_sql_script(sql_script, return_result = True))