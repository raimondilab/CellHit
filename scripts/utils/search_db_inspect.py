import sys
sys.path.append('./../../')

from sqlalchemy import create_engine
from AsyncDistribJobs.operations import configure_database
from AsyncDistribJobs.operations import print_summary

from pathlib import Path

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze database statistics.")
    parser.add_argument("--database_path", help="Path to the database file (e.g., sqlite:///jobs.db)", type=str, default='./../../results/gdsc/search_database/gdsc_all_genes_search_database.db')
    args = parser.parse_args()

    database_path = Path(args.database_path)

    #check whether the database exists
    assert database_path.exists(), f"Database file not found at {database_path}"

    engine = create_engine(f'sqlite:///{args.database_path}')
    configure_database(engine)

    print_summary()