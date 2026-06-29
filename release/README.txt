# TranslateBook Windows Executable

## Quick Start

1. Extract TranslateBook.exe to a folder of your choice
2. Double-click TranslateBook.exe to start the server
3. Open your browser to http://localhost:5000
4. Choose your LLM provider (see below)

## LLM Providers

You need at least one LLM provider to translate:

- DeepSeek - Chinese AI lab: https://platform.deepseek.com/api_keys
- Gemini - Google AI: https://aistudio.google.com/apikey
- Mistral - French AI lab: https://console.mistral.ai/api-keys
- Ollama (Local) - Free, runs on your machine: https://ollama.com
- OpenAI - GPT models: https://platform.openai.com/api-keys
- OpenRouter - 200+ cloud models: https://openrouter.ai/keys
- Poe - Multi-model cloud aggregator: https://poe.com/api_key

For local translation with Ollama:
1. Install Ollama from https://ollama.com
2. Download a model: ollama pull qwen3:14b

## Choosing the Best Model for Your Language

Different models perform better for different target languages!

See our comprehensive benchmarks to find the best model:
https://github.com/hydropix/TranslateBooksWithLLMs/wiki

## First Run

On first run, the application will:
- Create a TranslateBook_Data folder next to the executable
- Generate a default .env configuration file
- Create necessary subdirectories (translated_files, checkpoints)

## Configuration

Edit TranslateBook_Data\.env to customize:
- LLM provider and model selection
- API keys for cloud providers
- Server port and host

## Usage

- Web UI: http://localhost:5000
- Supported formats: .txt, .epub, .srt, .docx, .odt
- Output files: TranslateBook_Data\translated_files\

## Links

- Full Documentation: https://github.com/hydropix/TranslateBooksWithLLMs
- Model Benchmarks: https://github.com/hydropix/TranslateBooksWithLLMs/wiki
- Report Issues: https://github.com/hydropix/TranslateBooksWithLLMs/issues
- OpenRouter Models: https://openrouter.ai/models
