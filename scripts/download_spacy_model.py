"""Script to download the spaCy en_core_web_sm pipeline."""

import subprocess
import sys


def main() -> None:
    print("Downloading spaCy en_core_web_sm model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully installed en_core_web_sm.")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
