"""Target-language packs.

Each subpackage holds the rules, vocabulary and prompt text that only make
sense for one target language. Pipeline code looks a pack up through the
language profile code (see src.utils.language_profiles) instead of carrying
language-specific literals itself.
"""
