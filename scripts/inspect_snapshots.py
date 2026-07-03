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

print(f"Total chunk_data rows: {len(rows)}")
hagley_snap_count = 0

for cidx, cdata_raw in rows:
    try:
        data = json.loads(cdata_raw)
        snap = data.get("context_snapshot")
        if snap:
            decoded, global_lore, dynamic_state = decode_context_snapshot(snap)
            if "Hagley" in decoded:
                hagley_snap_count += 1
                print(
                    f"Chunk {cidx} context_snapshot HAS Hagley (len: {len(decoded)})"
                )
                for line in decoded.splitlines():
                    if "Hagley" in line:
                        print(f"   {line}")
    except Exception as e:
        print(f"Error chunk {cidx}: {e}")

print(f"Total Hagley snapshots: {hagley_snap_count}")
