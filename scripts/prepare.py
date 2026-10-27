
from pathlib import Path

import duckdb

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True,
)

con = duckdb.connect()

# Load and validate raw data
con.execute("""
    CREATE TEMP TABLE clean AS
    SELECT *
    FROM read_csv_auto('data/raw/dataset.csv')
    WHERE label IS NOT NULL
""")

# Deterministic split for this demonstration
splits = {
    "train": "row_id % 10 < 6",
    "val": "row_id % 10 in (6, 7)",
    "test": "row_id % 10 >= 8",
}

for name, condition in splits.items():
    output = f"data/processed/{name}.parquet"

    con.execute(f"""
        COPY (
            SELECT
                f0, f1, f2, f3,
                f4, f5, f6, f7,
                label
            FROM clean
            WHERE {condition}
        )
        TO '{output}'
        (FORMAT PARQUET)
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output}')"
    ).fetchone()[0]

    print(f"{name}: {count} samples")

con.close()
