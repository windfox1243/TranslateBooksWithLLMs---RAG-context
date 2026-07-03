import re


def parse_and_convert_addressing_line(line: str) -> str:
    line_clean = line.strip()
    if not line_clean.startswith("-"):
        return line

    # Match format: - Speaker → Addressee: "Form1" | "Form2" | Description
    # or already has self-reference
    if "self-reference:" in line_clean:
        # Normalize any ta-cậu to ta-ngươi or tôi-cậu if needed, or leave clean
        return line

    pattern = r"^-\s*([^→:]+)\s*→\s*([^:]+):\s*\"([^\"]*)\"\s*\|\s*\"([^\"]*)\"\s*\|\s*(.*)$"
    match = re.match(pattern, line_clean)
    if not match:
        return line

    speaker = match.group(1).strip()
    addressee = match.group(2).strip()
    eng_form = match.group(3).strip()
    viet_form = match.group(4).strip()
    desc = match.group(5).strip()

    # Determine self-reference, second-person, vocative based on viet_form and relationships
    self_ref = ""
    second_person = ""
    vocative = viet_form

    viet_lower = viet_form.lower()
    desc_lower = desc.lower()

    # Rule deductions for Vietnamese address pairs
    if "mẹ" in desc_lower and "con" in desc_lower:
        if "mẹ với con" in desc_lower or "father to son" in desc_lower:
            self_ref = "mẹ"
            second_person = "con"
        elif "con với mẹ" in desc_lower:
            self_ref = "con"
            second_person = "mẹ"
    elif "cha" in desc_lower or "bố" in desc_lower:
        if "cha với con" in desc_lower or "father to" in desc_lower:
            self_ref = "cha"
            second_person = "con"
        elif "con với cha" in desc_lower:
            self_ref = "con"
            second_person = "cha"
    elif "anh" in desc_lower and "em" in desc_lower:
        if "anh trai với em" in desc_lower or "older brother" in desc_lower:
            self_ref = "anh"
            second_person = "em"
        elif "em trai với anh" in desc_lower or "younger brother" in desc_lower:
            self_ref = "em"
            second_person = "anh"
    elif "chị" in desc_lower and "em" in desc_lower:
        if "chị với em" in desc_lower or "older sister" in desc_lower:
            self_ref = "chị"
            second_person = "em"
        elif "em với chị" in desc_lower or "younger sister" in desc_lower:
            self_ref = "em"
            second_person = "chị"
    elif "tiền bối với hậu bối" in desc_lower or "senior" in desc_lower:
        self_ref = "anh" if "nam" in desc_lower else "tôi"
        second_person = "em"
    elif "hậu bối với tiền bối" in desc_lower or "junior" in desc_lower:
        self_ref = "em"
        second_person = "anh" if "nam" in desc_lower else "chị"
    elif (
        "bạn học" in desc_lower
        or "bạn thân" in desc_lower
        or "classmate" in desc_lower
        or "peer" in desc_lower
    ):
        if (
            "thân mật" in desc_lower
            or "bạn thân" in desc_lower
            or "informal" in desc_lower
        ):
            self_ref = "tớ"
            second_person = "cậu"
        else:
            self_ref = "tôi"
            second_person = "cậu"
    elif (
        "thù địch" in desc_lower
        or "kẻ thù" in desc_lower
        or "hostile" in desc_lower
    ):
        if "khinh miệt" in desc_lower or "bề trên" in desc_lower:
            self_ref = "ta"
            second_person = "ngươi"
        else:
            self_ref = "tôi"
            second_person = "ông" if "male" in desc_lower else "cô"
    elif (
        "hoàng đế" in desc_lower
        or "empress" in desc_lower
        or "thần dân" in desc_lower
    ):
        if "hoàng đế với" in desc_lower or "empress to" in desc_lower:
            self_ref = "ta"
            second_person = "ngươi"
        else:
            self_ref = "tôi"
            second_person = "Bệ hạ"
    elif "thần linh" in desc_lower or "deity" in desc_lower:
        if "thần linh với" in desc_lower:
            self_ref = "ta"
            second_person = "ngươi"
        else:
            self_ref = "tôi"
            second_person = "ngài"

    # Default fallback if unknown
    if not self_ref:
        if viet_lower in ["anh", "chị", "em", "thầy", "cô", "bệ hạ", "ngài"]:
            self_ref = "tôi"
            second_person = viet_lower
        else:
            self_ref = "tôi"
            second_person = "cậu"

    if not vocative:
        vocative = viet_form if viet_form else addressee

    converted = f'- {speaker} → {addressee}: "{eng_form}" | "self-reference: {self_ref}; second-person pronoun: {second_person}; vocative/address form: {vocative}" | {desc}\n'
    return converted


# Test on sample lines
sample = '- Maid → Frondier De Roach: "Frondier-nim" | "Cậu Frondier" | tôn trọng, người hầu với chủ nhân'
print(parse_and_convert_addressing_line(sample))
