import sqlite3
import json

def main():
    context_path = r'F:\TranslateBook_Data\Novel_Contexts\The_Academy_Weapon_Replicator__Vietnamese__context.txt'
    with open(context_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Clean up context file to have only the single current active entry for Cain -> Frondier
    clean_lines = []
    skip_next = False
    for line in lines:
        if line.startswith('- Cain → Frondier De Roach'):
            # Replace with single current active entry
            clean_lines.append('- Cain → Frondier De Roach: "the Frondier" | "self-reference: tôi; second-person pronoun: anh; vocative/address form: anh/anh Frondier" | mềm mỏng, bối rối/trút giận vì thất vọng, coi Frondier là người định hướng\n')
        else:
            clean_lines.append(line)

    latest_context = "".join(clean_lines)

    # Save to context file
    with open(context_path, 'w', encoding='utf-8') as f:
        f.write(latest_context)
    print("Cleaned context file with single current active entry for Cain -> Frondier.")

    # Create early context text (where Cain -> Frondier was ta - ngươi)
    early_lines = []
    for line in clean_lines:
        if line.startswith('- Cain → Frondier De Roach:'):
            early_lines.append('- Cain → Frondier De Roach: "the Frondier" | "self-reference: ta; second-person pronoun: ngươi; vocative/address form: Frondier" | thù địch, khinh miệt, không có quan hệ thân thiết\n')
        elif line.startswith('- Frondier De Roach → Cain:'):
            early_lines.append('- Frondier De Roach → Cain: "Cain" | "self-reference: tôi; second-person pronoun: cô; vocative/address form: Cain" | trang trọng, xã giao, người lạ/đối thủ\n')
        else:
            early_lines.append(line)

    early_context = "".join(early_lines)

    # Update DB snapshots by chunk index
    db_path = r'F:\TranslateBook_Data\data\jobs.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT translation_id, chunk_index, chunk_data FROM checkpoint_chunks")
    rows = cursor.fetchall()

    updated_early = 0
    updated_latest = 0

    for tid, chunk_idx, chunk_data_str in rows:
        if chunk_data_str:
            try:
                data = json.loads(chunk_data_str)
                if chunk_idx < 150:
                    data['context_snapshot'] = early_context
                    updated_early += 1
                else:
                    data['context_snapshot'] = latest_context
                    updated_latest += 1
                new_str = json.dumps(data, ensure_ascii=False)
                cursor.execute("UPDATE checkpoint_chunks SET chunk_data = ? WHERE translation_id = ? AND chunk_index = ?", (new_str, tid, chunk_idx))
            except Exception as e:
                pass

    conn.commit()
    conn.close()
    print(f"Updated DB snapshots: {updated_early} early chunks (<150), {updated_latest} latest chunks (>=150).")

if __name__ == '__main__':
    main()
