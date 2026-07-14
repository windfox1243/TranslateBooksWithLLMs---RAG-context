import pytest

from src.utils.narrator_conformance import audit_narrator_conformance


FIRST_PERSON_SOURCE = (
    "I crossed the empty station alone.\n"
    "I remembered the race as rain reached the windows.\n"
    "I promised myself that I would keep moving."
)


def test_unit_12_third_person_and_thought_are_not_narrator_violations():
    audit = audit_narrator_conformance(
        source_text=(
            "She crossed the empty station alone.\n"
            "(I should keep moving, she thought.)\n"
            "She watched the rain reach the windows."
        ),
        target_text=(
            "Cô ấy một mình băng qua nhà ga vắng.\n"
            "(Mình phải tiếp tục, cô nghĩ.)\n"
            "Cô ấy nhìn mưa chạm vào cửa sổ."
        ),
        source_language="English",
        target_language="Vietnamese",
    )

    assert audit["status"] == "not_applicable"
    assert audit["violating_segments"] == []


def test_unit_13_detects_narrative_to_but_preserves_dialogue_pronouns():
    audit = audit_narrator_conformance(
        source_text=FIRST_PERSON_SOURCE,
        target_text=(
            "Tớ một mình băng qua nhà ga vắng.\n"
            "“Tớ sẽ đợi cậu,” Special Week nói.\n"
            "“Em cảm ơn chị,” cô ấy nói.\n"
            "Tớ nhớ cuộc đua khi mưa chạm vào cửa sổ."
        ),
        source_language="English",
        target_language="Vietnamese",
    )

    assert audit["status"] == "review_required"
    assert audit["reason_codes"] == ["narrator_self_reference_mismatch"]
    assert len(audit["violating_segments"]) == 2
    assert all(item["blocking"] is False for item in audit["violating_segments"])
    assert all(item["observed_form"] == "tớ" for item in audit["violating_segments"])
    assert all("Special Week" not in item["target_span"] for item in audit["violating_segments"])


def test_corrected_unit_13_passes_without_rewriting_dialogue():
    target = (
        "Tôi một mình băng qua nhà ga vắng.\n"
        "“Tớ sẽ đợi cậu,” Special Week nói.\n"
        "“Em cảm ơn chị,” cô ấy nói.\n"
        "Tôi nhớ cuộc đua khi mưa chạm vào cửa sổ."
    )
    audit = audit_narrator_conformance(
        source_text=FIRST_PERSON_SOURCE,
        target_text=target,
        source_language="English",
        target_language="Vietnamese",
    )

    assert audit["status"] == "pass"
    assert "“Tớ sẽ đợi cậu,”" in target
    assert "“Em cảm ơn chị,”" in target


@pytest.mark.parametrize("target_language", [
    "English", "French", "Spanish", "German", "Vietnamese", "Chinese",
    "Japanese", "Korean", "Arabic", "Russian", "Hindi", "Thai",
    "Italian", "Portuguese", "Dutch", "Polish", "Turkish",
])
@pytest.mark.parametrize("file_type", ["txt", "epub", "docx", "srt"])
def test_supported_language_and_format_contract(target_language, file_type):
    audit = audit_narrator_conformance(
        source_text=FIRST_PERSON_SOURCE,
        target_text="A structurally valid translated narrative.",
        source_language="English",
        target_language=target_language,
        file_type=file_type,
    )

    assert audit["status"] in {"pass", "not_applicable"}
    assert "validator_version" in audit


def test_srt_dialogue_is_excluded_but_voice_over_is_audited():
    target = "Tớ nhớ cơn mưa.\nTớ vẫn tiếp tục."
    ordinary = audit_narrator_conformance(
        source_text=FIRST_PERSON_SOURCE,
        target_text=target,
        source_language="English",
        target_language="Vietnamese",
        file_type="srt",
    )
    voice_over = audit_narrator_conformance(
        source_text=FIRST_PERSON_SOURCE,
        target_text=target,
        source_language="English",
        target_language="Vietnamese",
        file_type="srt",
        dialogue_attribution={"voice_over": True},
    )

    assert ordinary["status"] == "not_applicable"
    assert voice_over["status"] == "review_required"


def test_explicit_narrator_policy_makes_mismatch_blocking():
    audit = audit_narrator_conformance(
        source_text=FIRST_PERSON_SOURCE,
        target_text="Tớ qua ga.\nTớ nhớ mưa.\nTớ tiếp tục.",
        source_language="English",
        target_language="Vietnamese",
        explicit_override="tôi",
    )

    assert audit["status"] == "fail"
    assert all(item["blocking"] is True for item in audit["violating_segments"])


def test_contextual_third_person_form_is_not_a_narrator_mismatch():
    audit = audit_narrator_conformance(
        source_text="He crossed the station. He remembered the rain.",
        target_text="Anh ấy qua ga. Anh nhớ cơn mưa.",
        source_language="English",
        target_language="Vietnamese",
    )

    assert audit["status"] == "not_applicable"
    assert audit["violating_segments"] == []


def test_inline_dialogue_is_masked_before_narrator_audit():
    audit = audit_narrator_conformance(
        source_text="I stopped. \"I will wait,\" she said. I continued.",
        target_text="Tôi dừng lại. \"Tớ sẽ đợi,\" cô ấy nói. Tôi đi tiếp.",
        source_language="English",
        target_language="Vietnamese",
        explicit_override="tôi",
    )

    assert audit["status"] == "pass"
    assert audit["violating_segments"] == []
