import re

path = r"F:\TranslateBook_Data\Novel_Contexts\The_Academy_Weapon_Replicator__Vietnamese__context.txt"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_addressing = False
mismatches = []
unformatted = []

for i, line in enumerate(lines):
    if "## CURRENT ADDRESSING FORMS" in line:
        in_addressing = True
        continue
    if in_addressing and line.startswith("## "):
        in_addressing = False
        break
    if in_addressing and line.strip().startswith("-"):
        l = line.strip()
        if "self-reference:" in l:
            m_self = re.search(r"self-reference:\s*([^;|\"]+)", l)
            m_sec = re.search(r"second-person pronoun:\s*([^;|\"]+)", l)
            if m_self and m_sec:
                s_val = m_self.group(1).strip()
                sec_val = m_sec.group(1).strip()

                # Suspicious pairs check
                if s_val == "tôi" and sec_val == "ngươi":
                    mismatches.append((i + 1, l, s_val, sec_val, "tôi - ngươi"))
                elif s_val == "ta" and sec_val in [
                    "cậu",
                    "anh",
                    "bạn",
                    "chị",
                    "em",
                ]:
                    mismatches.append(
                        (i + 1, l, s_val, sec_val, f"ta - {sec_val}")
                    )
                elif s_val == "tớ" and sec_val in ["ngươi", "ngài", "ông"]:
                    mismatches.append(
                        (i + 1, l, s_val, sec_val, f"tớ - {sec_val}")
                    )
        else:
            unformatted.append((i + 1, l))

print(f"Total unformatted addressing entries: {len(unformatted)}")
for line_num, text in unformatted[:15]:
    print(f"Line {line_num}: {text}")

print(f"\nTotal mismatched addressing entries: {len(mismatches)}")
for line_num, text, s, sec, reason in mismatches:
    print(f"Line {line_num} [{reason}]: {text}")
