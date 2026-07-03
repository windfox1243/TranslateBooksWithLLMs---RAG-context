import sqlite3
import glob
import re

def main():
    # 1. Update Context File
    context_path = r'F:\TranslateBook_Data\Novel_Contexts\The_Academy_Weapon_Replicator__Vietnamese__context.txt'
    with open(context_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated = False
    new_lines = []
    for line in lines:
        if line.startswith('- Philly Terst → Frondier De Roach:'):
            new_line = '- Philly Terst → Frondier De Roach: "Frondier" | "self-reference: ta; second-person pronoun: cậu; third-person pronoun: cậu ấy/cậu Frondier; vocative/address form: Frondier" | trang trọng, hoàng đế với thần dân/cộng sự\n'
            new_lines.append(new_line)
            updated = True
        elif line.startswith('- Pascal Schilitz → Frondier De Roach:'):
            new_line = '- Pascal Schilitz → Frondier De Roach: "Frondier" | "self-reference: tôi; second-person pronoun: cậu; third-person pronoun: cậu ấy/cậu Frondier; vocative/address form: Frondier" | thân mật, cựu giáo viên với học sinh\n'
            new_lines.append(new_line)
            updated = True
        else:
            new_lines.append(line)

    if updated:
        with open(context_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("Updated context file with third-person pronouns for Philly and Pascal -> Frondier.")

    # 2. Fix Database Chunks
    db_path = r'F:\TranslateBook_Data\data\jobs.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    full_context = "".join(new_lines)

    cursor.execute("SELECT translation_id, chunk_index, translated_text FROM checkpoint_chunks WHERE translated_text LIKE '%Frondier dự đoán rằng Hoàng đế%'")
    rows = cursor.fetchall()
    
    for tid, chunk_idx, text in rows:
        fixed_text = text
        # Fix Pascal's speech to Philly: 'anh ấy' -> 'cậu ấy'
        fixed_text = fixed_text.replace(
            "Frondier dự đoán rằng Hoàng đế sẽ thay đổi và xã hội giai cấp sẽ sụp đổ, và anh ấy đã nói điều này với Cain. Anh ấy nghĩ nó sẽ xảy ra sau chiến tranh, nhưng vì nó không thực sự xảy ra, Cain đã tức giận với Frondier. Nói rằng anh ấy đã không giữ lời hứa.",
            "Frondier dự đoán rằng Hoàng đế sẽ thay đổi và xã hội giai cấp sẽ sụp đổ, và cậu ấy đã nói điều này với Cain. Cậu ấy nghĩ nó sẽ xảy ra sau chiến tranh, nhưng vì nó không thực sự xảy ra, Cain đã tức giận với Frondier. Nói rằng cậu ấy đã không giữ lời hứa."
        )
        # Fix Philly's thoughts: 'anh ấy' / 'anh' -> 'cậu ấy' / 'cậu'
        fixed_text = fixed_text.replace(
            "‘Cain là thành viên của Indus, nhóm bình dân, và việc Frondier đặc biệt nói với cô ấy điều đó cho thấy anh ấy muốn thứ gì đó từ cô ấy.’",
            "‘Cain là thành viên của Indus, nhóm bình dân, và việc Frondier đặc biệt nói với cô ấy điều đó cho thấy cậu ấy muốn thứ gì đó từ cô ấy.’"
        )
        fixed_text = fixed_text.replace(
            "chẳng phải Frondier là người đã góp phần vào sự hồi phục của Bartello sao? Anh là người đã trao cho Philly Trái tim Rồng và ngăn chặn chiến tranh.",
            "chẳng phải Frondier là người đã góp phần vào sự hồi phục của Bartello sao? Cậu là người đã trao cho Philly Trái tim Rồng và ngăn chặn chiến tranh."
        )
        fixed_text = fixed_text.replace(
            "tất cả những người từng ca ngợi anh là anh hùng có thể sẽ quay lưng lại với anh.",
            "tất cả những người từng ca ngợi cậu ấy là anh hùng có thể sẽ quay lưng lại với cậu ấy."
        )
        # Fix Philly's speech to Pascal: 'anh ấy' -> 'cậu ấy'
        fixed_text = fixed_text.replace(
            "Rằng Frondier là một ác ma và không phải người gốc gia tộc Roach. Rằng lý do anh ấy ngăn chặn Belphegor là─",
            "Rằng Frondier là một ác ma và không phải người gốc gia tộc Roach. Rằng lý do cậu ấy ngăn chặn Belphegor là─"
        )

        if fixed_text != text:
            cursor.execute("UPDATE checkpoint_chunks SET translated_text = ? WHERE translation_id = ? AND chunk_index = ?", (fixed_text, tid, chunk_idx))
            print(f"Updated chunk {chunk_idx} in DB (tid: {tid}).")

    conn.commit()

    # Re-sync snapshots in chunk_data JSON across all chunks in DB
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
    print("Re-synced all context snapshots in DB chunk_data.")
    conn.close()

    # 3. Update Text Files in translated_files
    files = glob.glob(r'F:\TranslateBook_Data\translated_files\*.txt')
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if "Frondier dự đoán rằng Hoàng đế sẽ thay đổi" in content:
            new_content = content.replace(
                "Frondier dự đoán rằng Hoàng đế sẽ thay đổi và xã hội giai cấp sẽ sụp đổ, và anh ấy đã nói điều này với Cain. Anh ấy nghĩ nó sẽ xảy ra sau chiến tranh, nhưng vì nó không thực sự xảy ra, Cain đã tức giận với Frondier. Nói rằng anh ấy đã không giữ lời hứa.",
                "Frondier dự đoán rằng Hoàng đế sẽ thay đổi và xã hội giai cấp sẽ sụp đổ, và cậu ấy đã nói điều này với Cain. Cậu ấy nghĩ nó sẽ xảy ra sau chiến tranh, nhưng vì nó không thực sự xảy ra, Cain đã tức giận với Frondier. Nói rằng cậu ấy đã không giữ lời hứa."
            ).replace(
                "‘Cain là thành viên của Indus, nhóm bình dân, và việc Frondier đặc biệt nói với cô ấy điều đó cho thấy anh ấy muốn thứ gì đó từ cô ấy.’",
                "‘Cain là thành viên của Indus, nhóm bình dân, và việc Frondier đặc biệt nói với cô ấy điều đó cho thấy cậu ấy muốn thứ gì đó từ cô ấy.’"
            ).replace(
                "chẳng phải Frondier là người đã góp phần vào sự hồi phục của Bartello sao? Anh là người đã trao cho Philly Trái tim Rồng và ngăn chặn chiến tranh.",
                "chẳng phải Frondier là người đã góp phần vào sự hồi phục của Bartello sao? Cậu là người đã trao cho Philly Trái tim Rồng và ngăn chặn chiến tranh."
            ).replace(
                "tất cả những người từng ca ngợi anh là anh hùng có thể sẽ quay lưng lại với anh.",
                "tất cả những người từng ca ngợi cậu ấy là anh hùng có thể sẽ quay lưng lại với cậu ấy."
            ).replace(
                "Rằng Frondier là một ác ma và không phải người gốc gia tộc Roach. Rằng lý do anh ấy ngăn chặn Belphegor là─",
                "Rằng Frondier là một ác ma và không phải người gốc gia tộc Roach. Rằng lý do cậu ấy ngăn chặn Belphegor là─"
            )
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated translated text file: {filepath}")

if __name__ == '__main__':
    main()
