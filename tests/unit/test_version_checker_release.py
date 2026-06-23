from src.utils import version_checker


def test_stable_release_is_newer_than_same_version_prerelease():
    assert version_checker._is_newer(
        "v1.4.12",
        "1.4.12-context-experiment.2",
    )


def test_prerelease_is_not_newer_than_same_stable_version():
    assert not version_checker._is_newer(
        "1.4.12-context-experiment.3",
        "1.4.12",
    )


def test_prerelease_iterations_sort_numerically():
    assert version_checker._is_newer(
        "1.4.12-context-experiment.10",
        "1.4.12-context-experiment.2",
    )


def test_numeric_prerelease_identifier_sorts_below_text_identifier():
    assert version_checker._is_newer("1.4.12-alpha", "1.4.12-1")


def test_update_checker_targets_this_release_repository():
    assert version_checker.GITHUB_REPO_OWNER == "windfox1243"
    assert version_checker.GITHUB_REPO_NAME == "TranslateBooksWithLLMs---RAG-context"
