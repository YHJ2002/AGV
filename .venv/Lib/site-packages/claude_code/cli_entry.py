#!/usr/bin/env python3
"""
Claude Code CLI Entry Point
"""

import asyncio
import logging
from .src.app import main

def entry_point():
    """Entry point for the claude command"""
    try:
        # Disable verbose logging
        logging.getLogger().setLevel(logging.WARNING)
        for module in ["httpx", "urllib3", "httpcore", "asyncio", "filelock"]:
            logging.getLogger(module).setLevel(logging.ERROR)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    entry_point()