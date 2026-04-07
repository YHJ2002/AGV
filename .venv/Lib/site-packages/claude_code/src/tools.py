#!/usr/bin/env python3
"""
Claude Code Tools Implementation
All tools from the original Claude Code with comprehensive functionality
"""

import os
import re
import json
import glob
import shutil
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
import aiofiles
import httpx
from bs4 import BeautifulSoup
import nbformat
from markdownify import markdownify
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class ClaudeCodeTools:
    """Implementation of all Claude Code tools"""
    
    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.session_todos: List[Dict] = []
        self._http_client = httpx.AsyncClient()
    
    async def close(self):
        """Close HTTP client"""
        await self._http_client.aclose()
    
    # File Operations
    async def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
        """Read file from filesystem with line numbers"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace_root / path
            
            if not path.exists():
                return ToolResult("", error=f"File does not exist: {file_path}")
            
            if path.is_dir():
                return ToolResult("", error=f"Path is a directory: {file_path}")
            
            # Handle binary files (images, etc.)
            if self._is_binary_file(path):
                return ToolResult(f"Binary file: {path}", metadata={"type": "binary", "size": path.stat().st_size})
            
            async with aiofiles.open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = await f.readlines()
            
            # Apply offset and limit
            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]
            
            # Format with line numbers (cat -n style)
            formatted_lines = []
            for i, line in enumerate(lines, start=offset + 1):
                # Truncate long lines
                if len(line) > 2000:
                    line = line[:2000] + "... [truncated]\\n"
                formatted_lines.append(f"{i:6d}\\t{line.rstrip()}")
            
            content = "\\n".join(formatted_lines)
            return ToolResult(content, metadata={"lines": len(lines), "total_lines": len(lines) + offset})
            
        except Exception as e:
            return ToolResult("", error=f"Error reading file: {str(e)}")
    
    async def write(self, file_path: str, content: str) -> ToolResult:
        """Write content to file (overwrites existing)"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace_root / path
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            return ToolResult(f"File written successfully: {path}")
            
        except Exception as e:
            return ToolResult("", error=f"Error writing file: {str(e)}")
    
    async def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
        """Edit file with exact string replacement"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace_root / path
            
            if not path.exists():
                return ToolResult("", error=f"File does not exist: {file_path}")
            
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if old_string == new_string:
                return ToolResult("", error="old_string and new_string are identical")
            
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = content.count(old_string)
            else:
                if content.count(old_string) > 1:
                    return ToolResult("", error="old_string appears multiple times; use replace_all=true or provide more context")
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1 if old_string in content else 0
            
            if replacements == 0:
                return ToolResult("", error="old_string not found in file")
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(new_content)
            
            return ToolResult(f"Successfully replaced {replacements} occurrence(s) in {path}")
            
        except Exception as e:
            return ToolResult("", error=f"Error editing file: {str(e)}")
    
    async def multiedit(self, file_path: str, edits: List[Dict[str, Any]]) -> ToolResult:
        """Perform multiple edits atomically"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace_root / path
            
            if not path.exists():
                return ToolResult("", error=f"File does not exist: {file_path}")
            
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            current_content = content
            applied_edits = 0
            
            for edit in edits:
                old_str = edit.get('old_string', '')
                new_str = edit.get('new_string', '')
                replace_all = edit.get('replace_all', False)
                
                if old_str == new_str:
                    return ToolResult("", error=f"Edit {applied_edits + 1}: old_string and new_string are identical")
                
                if replace_all:
                    if old_str in current_content:
                        current_content = current_content.replace(old_str, new_str)
                        applied_edits += 1
                    else:
                        return ToolResult("", error=f"Edit {applied_edits + 1}: old_string not found")
                else:
                    if current_content.count(old_str) > 1:
                        return ToolResult("", error=f"Edit {applied_edits + 1}: old_string appears multiple times")
                    elif old_str in current_content:
                        current_content = current_content.replace(old_str, new_str, 1)
                        applied_edits += 1
                    else:
                        return ToolResult("", error=f"Edit {applied_edits + 1}: old_string not found")
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(current_content)
            
            return ToolResult(f"Successfully applied {applied_edits} edits to {path}")
            
        except Exception as e:
            return ToolResult("", error=f"Error in multiedit: {str(e)}")
    
    # Search & Discovery
    async def glob(self, pattern: str, path: str = None) -> ToolResult:
        """Find files matching glob pattern"""
        try:
            search_path = Path(path) if path else self.workspace_root
            if not search_path.is_absolute():
                search_path = self.workspace_root / search_path
            
            # Use glob.glob for pattern matching
            full_pattern = str(search_path / pattern)
            matches = glob.glob(full_pattern, recursive=True)
            
            # Convert to relative paths and sort by modification time
            relative_matches = []
            for match in matches:
                rel_path = os.path.relpath(match, self.workspace_root)
                try:
                    mtime = os.path.getmtime(match)
                    relative_matches.append((rel_path, mtime))
                except OSError:
                    relative_matches.append((rel_path, 0))
            
            # Sort by modification time (newest first)
            relative_matches.sort(key=lambda x: x[1], reverse=True)
            result_paths = [path for path, _ in relative_matches]
            
            content = "\\n".join(result_paths) if result_paths else "No files found matching pattern"
            return ToolResult(content, metadata={"count": len(result_paths)})
            
        except Exception as e:
            return ToolResult("", error=f"Error in glob search: {str(e)}")
    
    async def grep(self, pattern: str, include: str = None, path: str = None) -> ToolResult:
        """Search file contents using regex"""
        try:
            search_path = Path(path) if path else self.workspace_root
            if not search_path.is_absolute():
                search_path = self.workspace_root / search_path
            
            matches = []
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            
            # Build file list based on include pattern
            if include:
                file_pattern = str(search_path / "**" / include)
                files_to_search = glob.glob(file_pattern, recursive=True)
            else:
                files_to_search = []
                for root, dirs, files in os.walk(search_path):
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for file in files:
                        if not file.startswith('.'):
                            files_to_search.append(os.path.join(root, file))
            
            for file_path in files_to_search:
                if os.path.isfile(file_path) and not self._is_binary_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if regex.search(content):
                                rel_path = os.path.relpath(file_path, self.workspace_root)
                                mtime = os.path.getmtime(file_path)
                                matches.append((rel_path, mtime))
                    except Exception:
                        continue
            
            # Sort by modification time
            matches.sort(key=lambda x: x[1], reverse=True)
            result_paths = [path for path, _ in matches]
            
            content = "\\n".join(result_paths) if result_paths else "No files found containing pattern"
            return ToolResult(content, metadata={"count": len(result_paths), "pattern": pattern})
            
        except Exception as e:
            return ToolResult("", error=f"Error in grep search: {str(e)}")
    
    async def ls(self, path: str, ignore: List[str] = None) -> ToolResult:
        """List files and directories"""
        try:
            target_path = Path(path)
            if not target_path.is_absolute():
                target_path = self.workspace_root / target_path
            
            if not target_path.exists():
                return ToolResult("", error=f"Path does not exist: {path}")
            
            if not target_path.is_dir():
                return ToolResult("", error=f"Path is not a directory: {path}")
            
            items = []
            ignore_patterns = ignore or []
            
            for item in sorted(target_path.iterdir()):
                # Check ignore patterns
                if any(item.match(pattern) for pattern in ignore_patterns):
                    continue
                
                if item.is_dir():
                    items.append(f"{item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"{item.name} ({size} bytes)")
            
            content = "\\n".join(items) if items else "Directory is empty"
            return ToolResult(content, metadata={"count": len(items)})
            
        except Exception as e:
            return ToolResult("", error=f"Error listing directory: {str(e)}")
    
    # Code Execution
    async def bash(self, command: str, timeout: int = 120) -> ToolResult:
        """Execute bash command"""
        try:
            # Security check - avoid dangerous commands
            dangerous_patterns = [
                r'rm\s+-rf\s+/',
                r'rm\s+-fr\s+/',
                r'dd\s+if=',
                r':\(\)\{.*;\}',  # Fork bomb
                r'mkfs\.',
                r'format\s+c:',
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return ToolResult("", error="Command blocked for security reasons")
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.workspace_root
            )
            
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
                output = stdout.decode('utf-8', errors='replace')
                
                # Truncate if too long
                if len(output) > 30000:
                    output = output[:30000] + "\\n... [output truncated]"
                
                return ToolResult(output, metadata={"exit_code": process.returncode})
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult("", error=f"Command timed out after {timeout} seconds")
            
        except Exception as e:
            return ToolResult("", error=f"Error executing command: {str(e)}")
    
    # Jupyter Notebooks
    async def notebook_read(self, notebook_path: str) -> ToolResult:
        """Read Jupyter notebook with cells and outputs"""
        try:
            path = Path(notebook_path)
            if not path.is_absolute():
                path = self.workspace_root / path
            
            if not path.exists():
                return ToolResult("", error=f"Notebook does not exist: {notebook_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            content_parts = []
            for i, cell in enumerate(nb.cells):
                content_parts.append(f"Cell {i} ({cell.cell_type}):")
                content_parts.append(cell.source)
                
                if cell.cell_type == 'code' and cell.outputs:
                    content_parts.append("Outputs:")
                    for output in cell.outputs:
                        if output.output_type == 'stream':
                            content_parts.append(output.text)
                        elif output.output_type == 'execute_result':
                            if 'text/plain' in output.data:
                                content_parts.append(output.data['text/plain'])
                        elif output.output_type == 'error':
                            content_parts.append(f"Error: {output.ename}: {output.evalue}")
                
                content_parts.append("")  # Empty line between cells
            
            return ToolResult("\\n".join(content_parts), metadata={"cell_count": len(nb.cells)})
            
        except Exception as e:
            return ToolResult("", error=f"Error reading notebook: {str(e)}")
    
    async def notebook_edit(self, notebook_path: str, cell_number: int, new_source: str,
                           cell_type: str = None, edit_mode: str = "replace") -> ToolResult:
        """Edit Jupyter notebook cell"""
        try:
            path = Path(notebook_path)
            if not path.is_absolute():
                path = self.workspace_root / path
            
            if not path.exists():
                return ToolResult("", error=f"Notebook does not exist: {notebook_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            if edit_mode == "insert":
                if cell_type is None:
                    return ToolResult("", error="cell_type required for insert mode")
                
                new_cell = nbformat.v4.new_code_cell(new_source) if cell_type == "code" else nbformat.v4.new_markdown_cell(new_source)
                nb.cells.insert(cell_number, new_cell)
                
            elif edit_mode == "delete":
                if 0 <= cell_number < len(nb.cells):
                    del nb.cells[cell_number]
                else:
                    return ToolResult("", error=f"Cell index {cell_number} out of range")
                    
            else:  # replace
                if 0 <= cell_number < len(nb.cells):
                    nb.cells[cell_number].source = new_source
                    if cell_type:
                        nb.cells[cell_number].cell_type = cell_type
                else:
                    return ToolResult("", error=f"Cell index {cell_number} out of range")
            
            with open(path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            return ToolResult(f"Successfully {edit_mode}d cell in notebook")
            
        except Exception as e:
            return ToolResult("", error=f"Error editing notebook: {str(e)}")
    
    # Web Operations
    async def web_fetch(self, url: str, prompt: str) -> ToolResult:
        """Fetch and analyze web content"""
        try:
            response = await self._http_client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            # Convert HTML to markdown
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
            
            # Convert to markdown
            markdown_content = markdownify(str(soup), heading_style="ATX")
            
            # Truncate if too long
            if len(markdown_content) > 10000:
                markdown_content = markdown_content[:10000] + "\\n... [content truncated]"
            
            # Simple analysis based on prompt (in a real implementation, this would use an LLM)
            analysis = f"Content fetched from {url}. Prompt: {prompt}\\n\\nContent:\\n{markdown_content}"
            
            return ToolResult(analysis, metadata={"url": url, "content_length": len(markdown_content)})
            
        except Exception as e:
            return ToolResult("", error=f"Error fetching web content: {str(e)}")
    
    async def web_search(self, query: str, allowed_domains: List[str] = None, 
                        blocked_domains: List[str] = None) -> ToolResult:
        """Search the web (placeholder implementation)"""
        # Note: This is a placeholder. In a real implementation, you'd integrate with a search API
        return ToolResult(
            f"Web search results for: {query}\\n\\n"
            "Note: Web search functionality requires API integration with a search provider.",
            metadata={"query": query}
        )
    
    # Task Management
    async def todo_read(self) -> ToolResult:
        """Read current todo list"""
        if not self.session_todos:
            return ToolResult("[]", metadata={"count": 0})
        
        return ToolResult(json.dumps(self.session_todos, indent=2), metadata={"count": len(self.session_todos)})
    
    async def todo_write(self, todos: List[Dict[str, Any]]) -> ToolResult:
        """Write/update todo list"""
        try:
            # Validate todo structure
            for todo in todos:
                required_fields = ["content", "status", "priority", "id"]
                if not all(field in todo for field in required_fields):
                    return ToolResult("", error=f"Todo missing required fields: {required_fields}")
                
                if todo["status"] not in ["pending", "in_progress", "completed", "cancelled"]:
                    return ToolResult("", error=f"Invalid status: {todo['status']}")
                
                if todo["priority"] not in ["high", "medium", "low"]:
                    return ToolResult("", error=f"Invalid priority: {todo['priority']}")
            
            self.session_todos = todos
            return ToolResult(f"Updated todo list with {len(todos)} items")
            
        except Exception as e:
            return ToolResult("", error=f"Error updating todos: {str(e)}")
    
    # Task/Agent functionality
    async def task(self, description: str, prompt: str) -> ToolResult:
        """Launch autonomous agent for complex tasks"""
        # This is a placeholder - in the real implementation, this would launch a sub-agent
        return ToolResult(
            f"Task: {description}\\n\\nThis would launch an autonomous agent with the following prompt:\\n{prompt}\\n\\n"
            "Note: Task agent functionality would be implemented with full tool access.",
            metadata={"description": description}
        )
    
    # Utility methods
    def _is_binary_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\\0' in chunk
        except:
            return True

# Tool registry for easy access
CLAUDE_TOOLS = {
    "read": "Read file from filesystem with line numbers",
    "write": "Write content to file (overwrites existing)",
    "edit": "Edit file with exact string replacement", 
    "multiedit": "Perform multiple edits atomically",
    "glob": "Find files matching glob pattern",
    "grep": "Search file contents using regex",
    "ls": "List files and directories",
    "bash": "Execute bash command",
    "notebook_read": "Read Jupyter notebook with cells and outputs",
    "notebook_edit": "Edit Jupyter notebook cell",
    "web_fetch": "Fetch and analyze web content",
    "web_search": "Search the web",
    "todo_read": "Read current todo list",
    "todo_write": "Write/update todo list",
    "task": "Launch autonomous agent for complex tasks"
}