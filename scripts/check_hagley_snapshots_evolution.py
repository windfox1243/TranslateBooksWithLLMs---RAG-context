import json
import sqlite3
import sys

sys.path.insert(0, ".")
from src.utils.novel_context import decode_context_snapshot

db_path = r"F:\TranslateBook_Data\data\jobs.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute(
    "SELECT chunk_index, chunk_data FROM checkpoint_chunks WHERE translation_id = 'trans_1782968092354' AND chunk_data IS NOT NULL"
)
rows = c.fetchall()

found_evolutions = []
for cidx, cdata_raw in rows:
    try:
        data = json.loads(cdata_raw)
        snap = data.get("context_snapshot")
        if snap:
            decoded, _, _ = decode_context_snapshot(
                snap, canonicalize_full_snapshot=False
            )
            if "Hagley" in decoded:
                lines = decoded.splitlines()
                for line in lines:
                    if "Hagley" in line and any(
                        kw in line
                        for kw in [
                            "↔",
                            "→",
                            "imprisoned",
                            "Morion",
                            "kẻ thù",
                            "thù địch",
                            "Selena",
                            "Belphegor",
                            "Azier",
                            "Yeolgot",
                        ]
                    ):
                        found_evolutions.append((cidx, line))
    except Exception as e:
        pass

print(
    f"Found {len(found_evolutions)} Hagley dynamic lines across chunk snapshots:"
)
seen = set()
for cidx, line in found_evolutions:
    if line not in seen:
        seen.add(line)
        print(f"Chunk {cidx}: {line}")
