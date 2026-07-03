import base64
import json
import os
import sqlite3
import sys
import zlib

sys.path.insert(0, ".")
from src.utils.novel_context import decode_context_snapshot

db_path = r"F:\TranslateBook_Data\data\jobs.db"
translated_dir = r"F:\TranslateBook_Data\translated_files"

# 1. Update jobs.db
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute(
        "SELECT translation_id, chunk_index, chunk_data FROM checkpoint_chunks WHERE chunk_data IS NOT NULL"
    )
    rows = c.fetchall()

    db_updated_count = 0

    for tid, cidx, cdata_raw in rows:
        try:
            data = json.loads(cdata_raw)
            snap = data.get("context_snapshot")
            if snap:
                decoded, global_lore, dynamic_state = decode_context_snapshot(
                    snap, canonicalize_full_snapshot=False
                )
                original_decoded = decoded

                # Apply replacements
                if "serves Frondier" in decoded and "Hagley:" in decoded:
                    decoded = decoded.replace(
                        "- Hagley: Male, member of the Manggot group who serves Frondier, currently imprisoned in Morion.",
                        "- Hagley: Male, high-ranking member of the Manggot group and former superior of Selena, currently imprisoned in Morion.",
                    )

                if "Frondier De Roach → Hagley:" in decoded:
                    # Replace any variation of Frondier -> Hagley addressing entry
                    lines = decoded.splitlines()
                    new_lines = []
                    for line in lines:
                        if line.startswith("- Frondier De Roach → Hagley:"):
                            line = '- Frondier De Roach → Hagley: "Hagley" | "self-reference: ta; second-person pronoun: ngươi; vocative/address form: Hagley" | thù địch/đối đầu, kẻ thù'
                        new_lines.append(line)
                    decoded = "\n".join(new_lines)

                if decoded != original_decoded:
                    # Re-compress
                    compressed = base64.b64encode(
                        zlib.compress(decoded.encode("utf-8"))
                    ).decode("utf-8")
                    data["context_snapshot"] = compressed
                    new_cdata_raw = json.dumps(data, ensure_ascii=False)

                    c.execute(
                        "UPDATE checkpoint_chunks SET chunk_data = ? WHERE translation_id = ? AND chunk_index = ?",
                        (new_cdata_raw, tid, cidx),
                    )
                    db_updated_count += 1
        except Exception as e:
            print(f"Error updating chunk {cidx} in job {tid}: {e}")

    conn.commit()
    print(f"Updated {db_updated_count} context snapshots in jobs.db")

# 2. Update translated files in translated_files directory
txt_updated_files = 0
txt_replaced_lines = 0

if os.path.exists(translated_dir):
    for fname in os.listdir(translated_dir):
        if fname.endswith(".txt"):
            fpath = os.path.join(translated_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            modified = False
            new_lines = []

            for i, line in enumerate(lines):
                orig_line = line
                # Check for mismatched tôi-ngươi or incorrect master references in Hagley context
                # E.g. Frondier speaking to Hagley: "Tôi là Hagley..." -> Hagley speaking, fine.
                # If Frondier speaks to Hagley with "tôi" and "ngươi":
                if "ngươi" in line and "tôi" in line:
                    # If this is Frondier's dialogue to Hagley or in general
                    # e.g. "Ngươi là ai. Tại sao ngươi lại ở bên cạnh tôi?" -> "Ngươi là ai. Tại sao ngươi lại ở bên cạnh ta?"
                    if "ở bên cạnh tôi" in line:
                        line = line.replace(
                            "ở bên cạnh tôi", "ở bên cạnh ta"
                        )
                    if "kẻ thù của tôi" in line:
                        line = line.replace("kẻ thù của tôi", "kẻ thù của ta")

                if line != orig_line:
                    modified = True
                    txt_replaced_lines += 1
                new_lines.append(line)

            if modified:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                txt_updated_files += 1
                print(f"Updated file: {fname}")

print(
    f"Completed translated files update: {txt_updated_files} files, {txt_replaced_lines} lines updated."
)
