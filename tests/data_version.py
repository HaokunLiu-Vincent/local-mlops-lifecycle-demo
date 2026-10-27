
from pathlib import Path
import duckdb

# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

con = duckdb.connect()

for split in ["train", "val", "test"]:
    path = PROJECT_ROOT / "data" / "processed" / f"{split}.parquet"

    count = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(path)]
    ).fetchone()[0]

    print(f"{split}: {count}")

con.close()
