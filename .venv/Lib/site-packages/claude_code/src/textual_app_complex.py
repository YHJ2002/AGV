#!/usr/bin/env python3
"""
Claude Code - Textual-based UI Application
A modern, Claude-aesthetic interface using Textual framework
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Input, RichLog, Static, Button, 
    DirectoryTree, DataTable, TabbedContent, TabPane,
    LoadingIndicator, ProgressBar, Label, Switch, Select
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive, var
from textual.validation import Function, ValidationResult, Validator
from textual.widget import Widget
from textual.css.query import NoMatches
from textual.events import Key, Click
from textual.geometry import Offset
from textual.color import Color
from textual.design import ColorSystem

from rich.console import RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table
from rich import box
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn

from .agent import ClaudeCodeAgent

# Claude-inspired design system
CLAUDE_DESIGN = {
    "primary": "#E67E50",        # Warm coral/orange from Claude branding
    "secondary": "#D4A574",      # Warm beige/tan  
    "accent": "#B8956A",         # Deeper warm tone
    "success": "#7B9E3F",        # Natural green
    "info": "#5B8AA6",           # Calm blue
    "warning": "#D49C3D",        # Warm amber
    "error": "#C85450",          # Warm red
    "text": "#2C2B29",           # Warm dark text
    "background": "#F5F5F2",     # Warm off-white background
    "surface": "#FEFEFE",        # Clean white surface
    "muted": "#8B8680",          # Muted text
    "border": "#E0DDD6",         # Soft border
}

class ClaudeInput(Input):
    """Custom input widget with Claude styling and auto-completion"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = [
            "/help", "/exit", "/status", "/clear", "/model", "/workspace",
            "/read", "/write", "/edit", "/ls", "/grep", "/bash", "/todo",
            "/chat", "/stream", "/interactive"
        ]
        self.files_cache = []
        
    def _get_file_suggestions(self, text: str) -> List[str]:
        """Get file suggestions based on input"""
        if text.startswith('@'):
            return [f"@{f}" for f in self.files_cache if text[1:].lower() in f.lower()][:10]
        return []
    
    def _get_command_suggestions(self, text: str) -> List[str]:
        """Get command suggestions based on input"""
        if text.startswith('/'):
            return [cmd for cmd in self.commands if cmd.startswith(text)][:10]
        return []

class StatusBar(Static):
    """Status bar showing current state and information"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.status = "ready"
        self.current_tool = None
        self.token_count = 0
        self.max_tokens = 4096
        
    def update_status(self, status: str, tool: Optional[str] = None):
        """Update the status display"""
        self.status = status
        self.current_tool = tool
        self.refresh()
        
    def render(self) -> RenderableType:
        """Render the status bar"""
        status_indicators = {
            "ready": "🟢",
            "processing": "🟡", 
            "error": "🔴",
            "waiting": "🔵"
        }
        
        indicator = status_indicators.get(self.status, "⚪")
        
        if self.status == "processing" and self.current_tool:
            status_text = f"{indicator} {self.current_tool.title()}..."
        else:
            status_text = f"{indicator} {self.status.title()}"
            
        token_info = f"Tokens: {self.token_count}/{self.max_tokens}"
        
        return f"[dim]{status_text}[/dim] • [dim]{token_info}[/dim]"

class ChatMessage(Static):
    """Individual chat message widget"""
    
    def __init__(self, content: str, role: str, timestamp: Optional[datetime] = None, **kwargs):
        super().__init__(**kwargs)
        self.content = content
        self.role = role  # "user", "assistant", "system", "tool"
        self.timestamp = timestamp or datetime.now()
        
    def render(self) -> RenderableType:
        """Render the chat message"""
        time_str = self.timestamp.strftime("%H:%M:%S")
        
        if self.role == "user":
            return Panel(
                self.content,
                title=f"[bold]You[/bold] • [dim]{time_str}[/dim]",
                border_style=CLAUDE_DESIGN["info"],
                box=box.ROUNDED,
                padding=(0, 1)
            )
        elif self.role == "assistant":
            return Panel(
                Markdown(self.content),
                title=f"[bold]Claude[/bold] • [dim]{time_str}[/dim]",
                border_style=CLAUDE_DESIGN["primary"],
                box=box.ROUNDED,
                padding=(0, 1)
            )
        elif self.role == "tool":
            return Panel(
                self.content,
                title=f"[bold]Tool[/bold] • [dim]{time_str}[/dim]",
                border_style=CLAUDE_DESIGN["success"],
                box=box.ROUNDED,
                padding=(0, 1)
            )
        else:
            return Panel(
                self.content,
                title=f"[bold]System[/bold] • [dim]{time_str}[/dim]",
                border_style=CLAUDE_DESIGN["muted"],
                box=box.ROUNDED,
                padding=(0, 1)
            )

class ChatLog(ScrollableContainer):
    """Chat log container with messages"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[ChatMessage] = []
        
    def add_message(self, content: str, role: str, timestamp: Optional[datetime] = None):
        """Add a new message to the chat"""
        message = ChatMessage(content, role, timestamp)
        self.messages.append(message)
        self.mount(message)
        self.scroll_end()
        
    def clear_messages(self):
        """Clear all messages"""
        for message in self.messages:
            message.remove()
        self.messages.clear()

class FileExplorer(Container):
    """File explorer panel"""
    
    def __init__(self, workspace_path: str, **kwargs):
        super().__init__(**kwargs)
        self.workspace_path = Path(workspace_path)
        
    def compose(self) -> ComposeResult:
        """Compose the file explorer"""
        yield Static(f"📁 {self.workspace_path.name}", classes="panel-title")
        yield DirectoryTree(str(self.workspace_path), id="file-tree")

class ToolOutput(ScrollableContainer):
    """Tool output display"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def show_tool_result(self, tool_name: str, result: dict, args: dict):
        """Show tool execution result"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if result.get("error"):
            status_icon = "🔴"
            status_text = "Failed"
            content = f"Error: {result['error']}"
        else:
            status_icon = "🟢"
            status_text = "Success"
            content = result.get("content", "Completed successfully")
            
        panel = Panel(
            content,
            title=f"{status_icon} {tool_name.title()} • {status_text} • {timestamp}",
            border_style=CLAUDE_DESIGN["success"] if not result.get("error") else CLAUDE_DESIGN["error"],
            box=box.ROUNDED
        )
        
        widget = Static(panel)
        self.mount(widget)
        self.scroll_end()

class SettingsScreen(ModalScreen):
    """Settings modal screen"""
    
    def compose(self) -> ComposeResult:
        """Compose the settings screen"""
        with Container(id="settings-container"):
            yield Static("⚙️ Settings", classes="modal-title")
            with Vertical():
                yield Label("Model Settings")
                yield Select(
                    [("Qwen 2.5", "qwen2.5:7b-instruct"), ("Gemini", "gemini-2.5-flash-preview-05-20")],
                    id="model-select"
                )
                yield Label("Provider")
                yield Select([("Ollama", "ollama"), ("Gemini", "gemini")], id="provider-select")
                yield Label("Enable Thinking")
                yield Switch(id="thinking-switch")
                with Horizontal():
                    yield Button("Save", variant="primary", id="save-settings")
                    yield Button("Cancel", variant="default", id="cancel-settings")
                    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "save-settings":
            # Save settings logic here
            self.dismiss(True)
        elif event.button.id == "cancel-settings":
            self.dismiss(False)

class HelpScreen(ModalScreen):
    """Help modal screen"""
    
    def compose(self) -> ComposeResult:
        """Compose the help screen"""
        help_content = """
# Claude Code Commands

## Chat Commands
- Just type your message to chat with Claude
- Use `@filename` to add files to context
- Use `/` commands for specific actions

## Available Commands
- `/help` - Show this help
- `/exit` - Exit Claude Code
- `/status` - Show system status
- `/clear` - Clear conversation
- `/model` - Show current model
- `/workspace` - Show workspace info
- `/read <file>` - Read a file
- `/write <file>` - Write to a file
- `/edit <file>` - Edit a file
- `/ls <path>` - List directory
- `/grep <pattern>` - Search files
- `/bash <command>` - Execute command
- `/todo` - Manage todo list

## Keyboard Shortcuts
- `Ctrl+C` - Exit
- `Ctrl+H` - Show help
- `Ctrl+S` - Settings
- `F1` - Toggle file explorer
- `F2` - Toggle tool output
- `Esc` - Cancel current operation
        """
        
        with Container(id="help-container"):
            yield Static("📖 Help", classes="modal-title")
            yield Static(Markdown(help_content), classes="help-content")
            yield Button("Close", variant="primary", id="close-help")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press"""
        if event.button.id == "close-help":
            self.dismiss()

class ClaudeCodeTUI(App):
    """Main Textual application for Claude Code"""
    
    CSS = """
    /* Claude-inspired design system */
    Screen {
        background: #F5F5F2;
    }
    
    .claude-header {
        background: #E67E50;
        color: white;
        text-align: center;
        height: 3;
        content-align: center middle;
    }
    
    .claude-footer {
        background: #D4A574;
        color: white;
        height: 1;
    }
    
    .main-container {
        layout: horizontal;
        height: 1fr;
    }
    
    .left-panel {
        width: 25%;
        background: #FEFEFE;
        border-right: solid #E0DDD6;
    }
    
    .center-panel {
        width: 1fr;
        background: #F5F5F2;
        layout: vertical;
    }
    
    .right-panel {
        width: 30%;
        background: #FEFEFE;
        border-left: solid #E0DDD6;
    }
    
    .chat-container {
        height: 1fr;
        background: #FEFEFE;
        border: solid #E0DDD6;
        margin: 1;
        padding: 1;
    }
    
    .input-container {
        height: 3;
        background: #FEFEFE;
        border: solid #E0DDD6;
        margin: 1;
        padding: 1;
    }
    
    .panel-title {
        background: #E67E50;
        color: white;
        text-align: center;
        height: 1;
        content-align: center middle;
        text-style: bold;
    }
    
    .status-bar {
        background: #D4A574;
        color: white;
        height: 1;
        padding: 0 1;
        content-align: left middle;
    }
    
    .modal-title {
        text-align: center;
        text-style: bold;
        color: #E67E50;
        margin-bottom: 1;
    }
    
    .help-content {
        height: 1fr;
        background: #FEFEFE;
        border: solid #E0DDD6;
        padding: 1;
    }
    
    #settings-container {
        width: 60;
        height: 30;
        background: #FEFEFE;
        border: solid #E67E50;
        padding: 2;
    }
    
    #help-container {
        width: 80;
        height: 40;
        background: #FEFEFE;
        border: solid #E67E50;
        padding: 2;
    }
    
    Input {
        border: solid #E0DDD6;
    }
    
    Input:focus {
        border: solid #E67E50;
    }
    
    Button {
        margin: 1;
    }
    
    Button.-primary {
        background: #E67E50;
        color: white;
    }
    
    Button.-success {
        background: #7B9E3F;
        color: white;
    }
    
    Button.-warning {
        background: #D49C3D;
        color: white;
    }
    
    Button.-error {
        background: #C85450;
        color: white;
    }
    
    DirectoryTree {
        background: #FEFEFE;
    }
    
    LoadingIndicator {
        color: #E67E50;
    }
    
    .processing {
        color: #D49C3D;
    }
    
    .success {
        color: #7B9E3F;
    }
    
    .error {
        color: #C85450;
    }
    """
    
    TITLE = "Claude Code - AI Development Assistant"
    SUB_TITLE = "Powered by Textual"
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+h", "help", "Help"),
        Binding("ctrl+s", "settings", "Settings"), 
        Binding("f1", "toggle_files", "Files"),
        Binding("f2", "toggle_tools", "Tools"),
        Binding("escape", "cancel", "Cancel"),
    ]
    
    show_files = reactive(True)
    show_tools = reactive(True)
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 host: str = "http://192.168.170.76:11434",
                 workspace_path: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 num_ctx: int = 4096,
                 enable_thinking: bool = False,
                 provider: str = "ollama",
                 api_key: Optional[str] = None,
                 verbose: bool = False):
        super().__init__()
        
        # App configuration
        self.workspace_path = Path(workspace_path).resolve() if workspace_path else Path.cwd()
        self.model_name = model_name or ("gemini-2.5-flash-preview-05-20" if provider == "gemini" else "qwen2.5:7b-instruct")
        self.host = host
        self.num_ctx = num_ctx
        self.enable_thinking = enable_thinking
        self.provider = provider
        self.api_key = api_key
        self.verbose = verbose
        
        # Initialize agent
        self.agent = ClaudeCodeAgent(
            model_name=self.model_name,
            host=self.host,
            workspace_root=str(self.workspace_path),
            system_prompt=system_prompt,
            num_ctx=self.num_ctx,
            enable_thinking=self.enable_thinking,
            provider=self.provider,
            api_key=self.api_key,
            status_callback=self._handle_agent_status
        )
        
        # Chat state
        self.files_in_context: Set[Path] = set()
        self.current_status = "ready"
        self.current_tool = None
        
    def compose(self) -> ComposeResult:
        """Compose the main UI"""
        yield Static("🤖 Claude Code - AI Development Assistant", classes="claude-header")
        
        with Container(classes="main-container"):
            # Left panel - File Explorer
            with Container(classes="left-panel", id="files-panel"):
                yield FileExplorer(str(self.workspace_path))
                
            # Center panel - Chat
            with Container(classes="center-panel"):
                with Container(classes="chat-container"):
                    yield ChatLog(id="chat-log")
                    
                with Container(classes="input-container"):
                    yield ClaudeInput(
                        placeholder="Type your message, /command, or @filename...",
                        id="chat-input"
                    )
                    
            # Right panel - Tool Output
            with Container(classes="right-panel", id="tools-panel"):
                yield Static("🔧 Tool Output", classes="panel-title")
                yield ToolOutput(id="tool-output")
        
        yield StatusBar(classes="status-bar", id="status-bar")
        
    def on_mount(self) -> None:
        """Called when the app is mounted"""
        self.query_one("#chat-log").add_message(
            "🤖 Welcome to Claude Code! Type `/help` for commands or just start chatting.",
            "system"
        )
        
        # Update status bar
        status_bar = self.query_one("#status-bar")
        status_bar.token_count = 0
        status_bar.max_tokens = self.num_ctx
        status_bar.update_status("ready")
        
    def watch_show_files(self, show_files: bool) -> None:
        """React to show_files changes"""
        self.query_one("#files-panel").display = show_files
        
    def watch_show_tools(self, show_tools: bool) -> None:
        """React to show_tools changes"""  
        self.query_one("#tools-panel").display = show_tools
        
    def action_help(self) -> None:
        """Show help screen"""
        self.push_screen(HelpScreen())
        
    def action_settings(self) -> None:
        """Show settings screen"""
        self.push_screen(SettingsScreen())
        
    def action_toggle_files(self) -> None:
        """Toggle file explorer"""
        self.show_files = not self.show_files
        
    def action_toggle_tools(self) -> None:
        """Toggle tool output"""
        self.show_tools = not self.show_tools
        
    def action_cancel(self) -> None:
        """Cancel current operation"""
        self.current_status = "ready"
        self.query_one("#status-bar").update_status("ready")
        
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id != "chat-input":
            return
            
        user_input = event.value.strip()
        if not user_input:
            return
            
        # Clear input
        event.input.value = ""
        
        # Add user message to chat
        chat_log = self.query_one("#chat-log")
        chat_log.add_message(user_input, "user")
        
        # Update status
        status_bar = self.query_one("#status-bar")
        status_bar.update_status("processing")
        
        # Handle @ file references
        if user_input.startswith('@'):
            file_path = user_input[1:].strip()
            await self._add_file_to_context(file_path)
            status_bar.update_status("ready")
            return
            
        # Handle commands
        if user_input.startswith('/'):
            await self._handle_command(user_input)
            status_bar.update_status("ready")
            return
            
        # Handle regular chat
        try:
            response = await self.agent.agent_response(user_input)
            chat_log.add_message(response, "assistant")
        except Exception as e:
            chat_log.add_message(f"Error: {str(e)}", "system")
            status_bar.update_status("error")
            await asyncio.sleep(2)
            
        status_bar.update_status("ready")
        
    async def _add_file_to_context(self, file_path: str):
        """Add file to context"""
        path = Path(file_path).resolve()
        chat_log = self.query_one("#chat-log")
        
        if path.exists() and path.is_file():
            self.files_in_context.add(path)
            chat_log.add_message(f"✅ Added {path.relative_to(self.workspace_path)} to context", "system")
        else:
            chat_log.add_message(f"❌ File not found: {file_path}", "system")
            
    async def _handle_command(self, command_input: str):
        """Handle slash commands"""
        parts = command_input.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        chat_log = self.query_one("#chat-log")
        
        if command == "/help":
            self.action_help()
        elif command == "/exit":
            self.exit()
        elif command == "/status":
            await self._show_status()
        elif command == "/clear":
            chat_log.clear_messages()
            chat_log.add_message("🗑️ Conversation cleared", "system")
        elif command == "/model":
            chat_log.add_message(f"📊 Current model: {self.model_name}", "system")
        elif command == "/workspace":
            chat_log.add_message(f"📁 Workspace: {self.workspace_path}", "system")
        elif command == "/read":
            if args:
                await self._read_file(args)
            else:
                chat_log.add_message("❌ Usage: /read <file_path>", "system")
        elif command == "/ls":
            await self._list_directory(args or ".")
        elif command == "/grep":
            if args:
                await self._grep_files(args)
            else:
                chat_log.add_message("❌ Usage: /grep <pattern>", "system")
        elif command == "/bash":
            if args:
                await self._run_bash(args)
            else:
                chat_log.add_message("❌ Usage: /bash <command>", "system")
        elif command == "/todo":
            await self._show_todos()
        else:
            chat_log.add_message(f"❌ Unknown command: {command}", "system")
            
    async def _show_status(self):
        """Show system status"""
        chat_log = self.query_one("#chat-log")
        status_content = f"""
**System Status**
- Workspace: `{self.workspace_path}`
- Model: `{self.model_name}`
- Provider: `{self.provider}`
- Host: `{self.host}`
- Context Size: `{self.num_ctx}`
- Files in Context: `{len(self.files_in_context)}`
- Thinking Mode: `{'Enabled' if self.enable_thinking else 'Disabled'}`
        """
        chat_log.add_message(status_content.strip(), "system")
        
    async def _read_file(self, file_path: str):
        """Read a file"""
        try:
            result = await self.agent.tools.read(file_path)
            tool_output = self.query_one("#tool-output")
            tool_output.show_tool_result("read", result.__dict__, {"file_path": file_path})
            
            if not result.error:
                chat_log = self.query_one("#chat-log")
                syntax = Syntax(result.content, "text", theme="monokai", line_numbers=True)
                chat_log.add_message(f"📄 **File: {file_path}**\n\n{result.content}", "system")
        except Exception as e:
            chat_log = self.query_one("#chat-log")
            chat_log.add_message(f"❌ Error reading file: {str(e)}", "system")
            
    async def _list_directory(self, path: str):
        """List directory contents"""
        try:
            result = await self.agent.tools.ls(path)
            tool_output = self.query_one("#tool-output")
            tool_output.show_tool_result("ls", result.__dict__, {"path": path})
            
            if not result.error:
                chat_log = self.query_one("#chat-log")
                chat_log.add_message(f"📁 **Directory: {path}**\n\n```\n{result.content}\n```", "system")
        except Exception as e:
            chat_log = self.query_one("#chat-log")
            chat_log.add_message(f"❌ Error listing directory: {str(e)}", "system")
            
    async def _grep_files(self, pattern: str):
        """Search files with grep"""
        try:
            result = await self.agent.tools.grep(pattern)
            tool_output = self.query_one("#tool-output")
            tool_output.show_tool_result("grep", result.__dict__, {"pattern": pattern})
            
            if not result.error:
                chat_log = self.query_one("#chat-log")
                chat_log.add_message(f"🔍 **Search: {pattern}**\n\n```\n{result.content}\n```", "system")
        except Exception as e:
            chat_log = self.query_one("#chat-log")
            chat_log.add_message(f"❌ Error searching files: {str(e)}", "system")
            
    async def _run_bash(self, command: str):
        """Run bash command"""
        try:
            result = await self.agent.tools.bash(command)
            tool_output = self.query_one("#tool-output")
            tool_output.show_tool_result("bash", result.__dict__, {"command": command})
            
            chat_log = self.query_one("#chat-log")
            if result.error:
                chat_log.add_message(f"💻 **Command: {command}**\n\n❌ Error: {result.error}", "system")
            else:
                chat_log.add_message(f"💻 **Command: {command}**\n\n```\n{result.content}\n```", "system")
        except Exception as e:
            chat_log = self.query_one("#chat-log")
            chat_log.add_message(f"❌ Error running command: {str(e)}", "system")
            
    async def _show_todos(self):
        """Show todo list"""
        try:
            result = await self.agent.tools.todo_read()
            if not result.error:
                todos_data = json.loads(result.content)
                chat_log = self.query_one("#chat-log")
                
                if not todos_data:
                    chat_log.add_message("📝 No todos found", "system")
                else:
                    todo_text = "📝 **Todo List**\n\n"
                    for todo in todos_data:
                        status_icon = "✅" if todo.get('status') == 'completed' else "⏳"
                        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(todo.get('priority', 'medium'), "🟡")
                        todo_text += f"{status_icon} {priority_icon} {todo.get('content', 'Unknown task')}\n"
                    
                    chat_log.add_message(todo_text.strip(), "system")
        except Exception as e:
            chat_log = self.query_one("#chat-log")
            chat_log.add_message(f"❌ Error reading todos: {str(e)}", "system")
            
    def _handle_agent_status(self, action: str, data):
        """Handle status updates from agent"""
        status_bar = self.query_one("#status-bar")
        tool_output = self.query_one("#tool-output")
        
        if action == "show_tool":
            status_bar.update_status("processing", data)
        elif action == "tool_completed":
            tool_output.show_tool_result(data["tool_name"], data["result"].__dict__, data["arguments"])
            status_bar.update_status("ready")
        elif action == "tool_error":
            tool_output.show_tool_result(data["tool_name"], {"error": data["error"]}, data.get("arguments", {}))
            status_bar.update_status("error")
            
    async def on_unmount(self) -> None:
        """Clean up when app is unmounted"""
        if hasattr(self.agent, 'close'):
            await self.agent.close()