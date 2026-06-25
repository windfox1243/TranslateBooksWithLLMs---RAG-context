# TranslateBook Windows Executable

Version: 1.4.17

## Quick Start

1. Extract TranslateBook.exe to a folder of your choice.
2. Double-click TranslateBook.exe to start the server.
3. Open your browser to http://localhost:5000.
4. Choose your LLM provider in the web UI.

## First Run

The app creates a TranslateBook_Data folder next to the executable, including a compact .env file, .env.example, Custom_Instructions, Novel_Contexts, and translated_files.

## Novel Context

This release includes bounded previous-source memory for context analysis through NOVEL_CONTEXT_SOURCE_MEMORY_CHARS. New compact .env files now include that setting by default, matching .env.example and the runtime default. It also blocks ambiguous narrative-role labels such as Protagonist from becoming canonical character identities.
