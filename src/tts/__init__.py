"""
TTS (Text-to-Speech) Module

Provides text-to-speech generation capabilities using Edge-TTS
with optional Opus encoding via ffmpeg.

Usage:
    from src.tts import TTSConfig, AudioProcessor, generate_tts_for_text

    # Simple usage
    success, message = await generate_tts_for_text(
        text="Hello, world!",
        output_path="output.opus",
        language="English"
    )

    # With custom config
    config = TTSConfig(
        voice="zh-CN-XiaoxiaoNeural",
        rate="+10%",
        output_format="opus"
    )
    processor = AudioProcessor(config)
    success, message = await processor.generate_audio(
        text="你好世界",
        output_path="chinese.opus",
        language="Chinese"
    )
"""
from .audio_processor import (
    AudioProcessor,
    check_ffmpeg_available,
    check_ffmpeg_with_instructions,
    chunk_text_for_tts,
    create_tts_provider,
    generate_tts_for_text,
    get_ffmpeg_install_instructions,
)
from .providers import (
    EdgeTTSProvider,
    ProgressCallback,
    TTSError,
    TTSProvider,
    TTSResult,
    VoiceInfo,
)
from .tts_config import (  # Environment variables
    DEFAULT_VOICES,
    TTS_BITRATE,
    TTS_ENABLED,
    TTS_OUTPUT_FORMAT,
    TTS_PROVIDER,
    TTS_RATE,
    TTS_VOICE,
    TTSConfig,
    get_voice_for_language,
)

__all__ = [
    # Config
    'TTSConfig',
    'DEFAULT_VOICES',
    'get_voice_for_language',
    'TTS_ENABLED',
    'TTS_PROVIDER',
    'TTS_VOICE',
    'TTS_RATE',
    'TTS_OUTPUT_FORMAT',
    'TTS_BITRATE',
    # Processor
    'AudioProcessor',
    'create_tts_provider',
    'generate_tts_for_text',
    'chunk_text_for_tts',
    'check_ffmpeg_available',
    'check_ffmpeg_with_instructions',
    'get_ffmpeg_install_instructions',
    # Providers
    'TTSProvider',
    'TTSResult',
    'VoiceInfo',
    'TTSError',
    'ProgressCallback',
    'EdgeTTSProvider',
]
