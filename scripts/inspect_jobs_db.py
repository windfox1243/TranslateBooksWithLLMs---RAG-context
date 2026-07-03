import json
import sqlite3

db_path = r"F:\TranslateBook_Data\data\jobs.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute(
    "SELECT translation_id, chunk_index, translated_text, chunk_data FROM checkpoint_chunks"
)
rows = c.fetchall()

print(f"Total checkpoint chunks: {len(rows)}")

matched_chunks = []

for tid, cidx, trans_text, chunk_data in rows:
    has_hagley_trans = "Hagley" in (trans_text or "")
    has_hagley_data = "Hagley" in (str(chunk_data) if chunk_data else "")
    has_serves_data = "serves Frondier" in (str(chunk_data) if chunk_data else "")
    if has_hagley_trans or has_hagley_data or has_serves_data:
        matched_chunks.append(
            (tid, cidx, has_hagley_trans, has_hagley_data, has_serves_data)
        )

print(f"Total matched chunks in jobs.db: {len(matched_chunks)}")
for tid, cidx, ht, hd, sd in matched_chunks[:20]:
    print(
        f"Job {tid}, Chunk {cidx}: in trans_text={ht}, in chunk_data={hd}, has_serves_data={sd}"
    )
