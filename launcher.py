"""
Launcher wrapper for PyInstaller executable
Handles proper working directory setup and .env file management
"""
import os
import sys
import shutil
from pathlib import Path

def setup_working_directory():
    """Setup proper working directory for the executable"""

    # Determine if running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        exe_dir = Path(sys.executable).parent

        # Create a data directory next to the executable
        app_data_dir = exe_dir / 'TranslateBook_Data'
        app_data_dir.mkdir(exist_ok=True)

        # Change working directory to app data folder
        os.chdir(app_data_dir)

        # Create necessary subdirectories
        (app_data_dir / 'translated_files').mkdir(exist_ok=True)
        (app_data_dir / 'checkpoints').mkdir(exist_ok=True)

        # Copy .env.example if it doesn't exist and is bundled
        bundle_dir = Path(sys._MEIPASS)
        env_example_path = app_data_dir / '.env.example'
        if not env_example_path.exists():
            bundled_env_example = bundle_dir / '.env.example'
            if bundled_env_example.exists():
                shutil.copy(bundled_env_example, env_example_path)

        # Seed Custom_Instructions folder with bundled examples on first run.
        # Skip if user already has the folder (preserves their custom files).
        custom_instructions_path = app_data_dir / 'Custom_Instructions'
        if not custom_instructions_path.exists():
            bundled_custom_instructions = bundle_dir / 'Custom_Instructions'
            if bundled_custom_instructions.exists():
                shutil.copytree(bundled_custom_instructions, custom_instructions_path)

        # Seed Novel_Contexts folder with default template if missing.
        novel_contexts_path = app_data_dir / 'Novel_Contexts'
        if not novel_contexts_path.exists():
            bundled_novel_contexts = bundle_dir / 'Novel_Contexts'
            if bundled_novel_contexts.exists():
                shutil.copytree(bundled_novel_contexts, novel_contexts_path)
            else:
                novel_contexts_path.mkdir(exist_ok=True)

        # Create default .env if it doesn't exist
        env_path = app_data_dir / '.env'
        if not env_path.exists():
            print("\n" + "="*70)
            print("FIRST RUN DETECTED")
            print("="*70)
            print("\nCreating default configuration file...")

            # Keep .env short and practical. The bundled .env.example remains
            # available next to it as the full commented reference.
            from src.utils.env_helper import write_compact_env
            write_compact_env(env_path)
            print(f"[OK] Configuration file created at: {env_path}")
            print("\n[INFO] You can edit this file later to customize settings")
            print("="*70)
            print()

        print(f"[INFO] Working directory: {app_data_dir}")
        print(f"[INFO] Translated files will be saved to: {app_data_dir / 'translated_files'}")
        print()

    else:
        # Running as normal Python script
        print("[DEV] Running as Python script (development mode)")

if __name__ == '__main__':
    try:
        # Setup environment
        setup_working_directory()

        # Import and start the server
        from translation_api import start_server
        start_server()

    except KeyboardInterrupt:
        print("\n\n[STOPPED] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print("\n" + "="*70)
        print("[ERROR] STARTUP ERROR")
        print("="*70)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
