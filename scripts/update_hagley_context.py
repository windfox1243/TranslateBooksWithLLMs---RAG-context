import os

path = r"F:\TranslateBook_Data\Novel_Contexts\The_Academy_Weapon_Replicator__Vietnamese__context.txt"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []

for line in lines:
    if "Frondier De Roach → Hagley:" in line:
        line = '- Frondier De Roach → Hagley: "Hagley" | "self-reference: ta; second-person pronoun: ngươi; vocative/address form: Hagley" | thù địch/đối đầu, kẻ thù\n'
    new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Updated line 1072 to ta-nguoi")
