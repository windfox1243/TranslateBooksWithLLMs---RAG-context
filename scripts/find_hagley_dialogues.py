import os

path = r"F:\TranslateBook_Data\translated_files\The_Academy_Weapon_Replicator (Vietnamese) (9).txt"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

dialogue_matches = []
for i, line in enumerate(lines):
    if "Hagley" in line or "hagley" in line:
        start = max(0, i - 8)
        end = min(len(lines), i + 9)
        for j in range(start, end):
            l = lines[j].strip()
            if ("“" in l or '"' in l) and any(
                kw in l.lower()
                for kw in ["tôi", "ngươi", "ta", "ngài", "cậu", "chủ nhân", "thiếu gia"]
            ):
                dialogue_matches.append((j + 1, l))

# Remove duplicates maintaining order
seen = set()
unique_matches = []
for idx, text in dialogue_matches:
    if idx not in seen:
        seen.add(idx)
        unique_matches.append((idx, text))

print(f"Found {len(unique_matches)} unique dialogue lines near Hagley:")
for idx, text in unique_matches[:40]:
    print(f"Line {idx}: {text}")
