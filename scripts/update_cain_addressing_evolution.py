import sqlite3
import json

def main():
    context_path = r'F:\TranslateBook_Data\Novel_Contexts\The_Academy_Weapon_Replicator__Vietnamese__context.txt'
    with open(context_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('- Frondier De Roach → Cain:'):
            new_lines.append('- Frondier De Roach → Cain: "Cain" | "self-reference: tôi; second-person pronoun: cô/em; vocative/address form: Cain" | khuyên bảo, bảo vệ, coi Cain là đứa trẻ cần cảm hóa\n')
        elif line.startswith('- Cain → Frondier De Roach:'):
            new_lines.append('- Cain → Frondier De Roach (Thù địch/Ban đầu): "the Frondier" | "self-reference: ta; second-person pronoun: ngươi; vocative/address form: Frondier" | thù địch, khinh miệt\n')
            new_lines.append('- Cain → Frondier De Roach (Sau khi cảm hóa/Nhắc lời hứa): "the Frondier" | "self-reference: tôi; second-person pronoun: anh; vocative/address form: anh/anh Frondier" | mềm mỏng, bối rối/trút giận vì thất vọng, coi Frondier là người định hướng\n')
        else:
            new_lines.append(line)

    full_context = "".join(new_lines)
    with open(context_path, 'w', encoding='utf-8') as f:
        f.write(full_context)
    print("Updated context file with Cain -> Frondier addressing evolution!")

    # Re-sync snapshots in DB chunk_data
    db_path = r'F:\TranslateBook_Data\data\jobs.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT translation_id, chunk_index, chunk_data FROM checkpoint_chunks")
    rows = cursor.fetchall()
    for tid, chunk_idx, chunk_data_str in rows:
        if chunk_data_str:
            try:
                data = json.loads(chunk_data_str)
                data['context_snapshot'] = full_context
                new_str = json.dumps(data, ensure_ascii=False)
                cursor.execute("UPDATE checkpoint_chunks SET chunk_data = ? WHERE translation_id = ? AND chunk_index = ?", (new_str, tid, chunk_idx))
            except Exception as e:
                pass
    conn.commit()
    conn.close()
    print("Re-synced all DB context snapshots successfully!")

if __name__ == '__main__':
    main()
