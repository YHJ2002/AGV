#!/usr/bin/env python3
"""
Claude Code - Main CLI Application
A Claude Code implementation with comprehensive tools and modern Textual UI
"""

import asyncio
import argparse
import logging
from pathlib import Path

# Configure logging to suppress INFO messages
logging.basicConfig(level=logging.WARNING)

# Silence standard library loggers
for logger_name in ["asyncio", "urllib", "urllib3", "filelock", "httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from .textual_app import ClaudeCodeSimple



    
    

async def main():
    """Main entry point"""
    # Disable verbose logging
    logging.getLogger().setLevel(logging.WARNING)
    for module in ["httpx", "urllib3", "httpcore", "asyncio"]:
        logging.getLogger(module).setLevel(logging.ERROR)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Claude Code - AI Development Assistant")
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--host", default="http://192.168.170.76:11434", help="Ollama host URL")
    parser.add_argument("-p", "--provider", default="ollama", choices=["ollama", "gemini"], help="LLM provider (default: ollama)")
    parser.add_argument("--api-key", default="AIzaSyBb8wTvVw9e25aX8XK-eBuu1JzDEPCdqUE", help="API key for Gemini provider")
    parser.add_argument("-w", "--workspace", help="Workspace directory path")
    parser.add_argument("-q", "--query", help="Initial query to process")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking process")
    parser.add_argument("--num-ctx", type=int, default=4096, help="Context window size")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode with token display")
    
    args = parser.parse_args()
    
    # Set default model based on provider if not specified
    if not args.model:
        if args.provider == "gemini":
            args.model = "gemini-2.5-flash-preview-05-20"
        else:  # ollama or default
            args.model = "qwen2.5:7b-instruct"
    
    # Create and run Textual app
    app = ClaudeCodeSimple(
        model_name=args.model,
        host=args.host,
        workspace_path=args.workspace,
        num_ctx=args.num_ctx,
        enable_thinking=args.thinking,
        provider=args.provider,
        api_key=args.api_key,
        verbose=args.verbose
    )
    
    await app.run_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")