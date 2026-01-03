#!/usr/bin/env python3
"""
Alternative method: Use yt-dlp to extract cookies from browser.

This method uses yt-dlp's built-in cookie extraction which may work
better than direct database access.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not output_file:
        backend_dir = Path(__file__).parent
        output_file = backend_dir / 'youtube_cookies.txt'
    else:
        output_file = Path(output_file)
    
    print("Using yt-dlp to extract cookies from Chrome...")
    print("This may take a moment...")
    
    # Use yt-dlp's cookie extraction
    # Format: yt-dlp --cookies-from-browser chrome --cookies output.txt "https://youtube.com"
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "yt_dlp",
                "--cookies-from-browser", "chrome",
                "--cookies", str(output_file),
                "--no-download",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Dummy video to trigger cookie extraction
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and output_file.exists():
            print(f"\nSuccess! Cookies saved to: {output_file}")
            print(f"\nTo use these cookies, add to your backend/.env file:")
            print(f"YOUTUBE_COOKIES={output_file.absolute()}")
        else:
            print("yt-dlp cookie extraction failed.")
            print(f"Error: {result.stderr}")
            print("\nTry the manual method instead (see extract_chrome_cookies.py for instructions)")
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        print("Cookie extraction timed out.")
        sys.exit(1)
    except FileNotFoundError:
        print("yt-dlp not found. Make sure it's installed:")
        print("  pip install yt-dlp")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

