import os

path = r"F:\TranslateBook_Data\translated_files\The_Academy_Weapon_Replicator (Vietnamese) (9).txt"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

mismatched = []
for i, line in enumerate(lines):
    if "Hagley" in line or "hagley" in line:
        start = max(0, i - 10)
        end = min(len(lines), i + 11)
        for j in range(start, end):
            l = lines[j].strip()
            if "“" in l or '"' in l:
                if (
                    "ngươi" in l.lower()
                    and ("tôi" in l.lower() or "mình" in l.lower())
                ):
                    mismatched.append((j + 1, l))

print(f"Found {len(mismatched)} mismatched lines:")
for idx, l in mismatched:
    print(f"Line {idx}: {l}")
