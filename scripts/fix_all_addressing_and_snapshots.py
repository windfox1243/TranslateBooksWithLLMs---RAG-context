import base64
import json
import os
import re
import sqlite3
import sys
import zlib

sys.path.insert(0, ".")
from src.utils.novel_context import decode_context_snapshot

context_path = r"F:\TranslateBook_Data\Novel_Contexts\The_Academy_Weapon_Replicator__Vietnamese__context.txt"
db_path = r"F:\TranslateBook_Data\data\jobs.db"


def clean_line(line: str) -> str:
    line_clean = line.strip()
    if not line_clean.startswith("-"):
        return line

    if "self-reference: ta; second-person pronoun: cậu" in line_clean:
        line_clean = line_clean.replace(
            "self-reference: ta; second-person pronoun: cậu",
            "self-reference: ta; second-person pronoun: ngươi",
        )

    # Convert remaining unformatted entries with slashes or quotes
    if "self-reference:" not in line_clean:
        pattern = r"^-\s*([^→:]+)\s*→\s*([^:]+):\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*)$"
        match = re.match(pattern, line_clean)
        if match:
            speaker = match.group(1).strip()
            addressee = match.group(2).strip()
            eng_form = match.group(3).strip().strip('"')
            viet_form = match.group(4).strip().strip('"')
            desc = match.group(5).strip()

            self_ref = "tôi"
            second_person = "cậu"
            vocative = viet_form if viet_form else addressee

            if "No direct address" in viet_form or "none" in viet_form.lower():
                self_ref = "tôi"
                second_person = "cậu"
                vocative = addressee

            line_clean = f'- {speaker} → {addressee}: "{eng_form}" | "self-reference: {self_ref}; second-person pronoun: {second_person}; vocative/address form: {vocative}" | {desc}'

    return line_clean + "\n"


if os.path.exists(context_path):
    with open(context_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_addressing = False
    new_context_lines = []

    for line in lines:
        if "## CURRENT ADDRESSING FORMS" in line:
            in_addressing = True
            new_context_lines.append(line)
            continue
        if in_addressing and line.startswith("## "):
            in_addressing = False

        if in_addressing and line.strip().startswith("-"):
            new_context_lines.append(clean_line(line))
        else:
            new_context_lines.append(line)

    with open(context_path, "w", encoding="utf-8") as f:
        f.writelines(new_context_lines)

    print("Final cleanup on context file completed.")

# Re-run db sync for all chunks
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT translation_id, chunk_index, chunk_data FROM checkpoint_chunks WHERE chunk_data IS NOT NULL"
    )
    rows = c.fetchall()

    db_updated = 0
    for tid, cidx, cdata_raw in rows:
        try:
            data = json.loads(cdata_raw)
            snap = data.get("context_snapshot")
            if snap:
                decoded, global_lore, dynamic_state = decode_context_snapshot(
                    snap, canonicalize_full_snapshot=False
                )
                orig_decoded = decoded

                lines = decoded.splitlines()
                new_snap_lines = []
                in_addr = False

                for line in lines:
                    if "## CURRENT ADDRESSING FORMS" in line:
                        in_addr = True
                        new_snap_lines.append(line)
                        continue
                    if in_addr and line.startswith("## "):
                        in_addr = False

                    if in_addr and line.strip().startswith("-"):
                        new_snap_lines.append(clean_line(line).strip())
                    else:
                        new_snap_lines.append(line)

                new_decoded = "\n".join(new_snap_lines)
                if new_decoded != orig_decoded:
                    compressed = base64.b64encode(
                        zlib.compress(new_decoded.encode("utf-8"))
                    ).decode("utf-8")
                    data["context_snapshot"] = compressed
                    c.execute(
                        "UPDATE checkpoint_chunks SET chunk_data = ? WHERE translation_id = ? AND chunk_index = ?",
                        (json.dumps(data, ensure_ascii=False), tid, cidx),
                    )
                    db_updated += 1
        except Exception:
            pass

    conn.commit()
    print(f"Final DB sync: updated {db_updated} snapshots.")
