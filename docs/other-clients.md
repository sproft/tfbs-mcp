# Connecting AI apps other than Claude

The [README](../README.md) covers Claude Code and Claude Desktop. This page is for everything else. Do steps 1 and 2 of the README first (create the environment, install, run `tfbs-mcp --check`), then come back here.

## Before you start

**Use the full path.** Your AI app is not started from your terminal, so it does not know about your environment, and the bare command `tfbs-mcp` will not be found. Every config below needs the absolute path that `tfbs-mcp --check` prints on its `executable:` line. The examples use `C:\Users\you\...` as a placeholder.

**How the app talks to the server.** The app starts `tfbs-mcp` itself and talks to it through the program's input and output (MCP calls this the *stdio* transport). The server does not open a network port. Apps that can only connect to a URL need a bridge, described under [ChatGPT and the Responses API](#chatgpt-and-the-responses-api).

**Editing JSON config files:**

- On Windows, write every backslash in a path twice: `C:\\Users\\you\\...`. Forward slashes also work: `C:/Users/you/...`.
- A JSON file has exactly one outer `{ }`. If the file already has a servers section, add the `"tfbs"` entry inside it, with a comma after the previous entry, instead of pasting a second block.
- Many of these files live in folders whose names start with a dot, which Windows Explorer will not create. Create them from the terminal, e.g. `mkdir $HOME\.gemini`.

## Code editors

| Editor | File | Top-level key |
|---|---|---|
| VS Code (Copilot agent mode) | `.vscode/mcp.json` in your project | **`servers`** |
| Cursor | `~/.cursor/mcp.json`, or `.cursor/mcp.json` per project | `mcpServers` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` (global only) | `mcpServers` |
| Cline | open via sidebar: **MCP Servers > Configure** | `mcpServers` |
| Zed | `settings.json` via **"zed: open settings file"** | **`context_servers`** |

VS Code uses `servers`, not `mcpServers`. A Cursor snippet pasted into VS Code does nothing:

```json
{
  "servers": {
    "tfbs": {
      "type": "stdio",
      "command": "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe",
      "args": []
    }
  }
}
```

Cursor, Windsurf and Cline use the same shape as Claude Desktop:

```json
{
  "mcpServers": {
    "tfbs": {
      "command": "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe",
      "args": []
    }
  }
}
```

Zed nests the same entry under `context_servers`. **Continue** uses `~/.continue/config.yaml`, which takes a list where each entry needs a `name`:

```yaml
mcpServers:
  - name: tfbs
    command: C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe
    args: []
```

## OpenAI and Google

| Product | Works with this server? |
|---|---|
| **OpenAI Codex** (CLI / IDE) | Yes, directly |
| **Gemini CLI** | Yes, directly |
| **OpenAI Agents SDK** (Python) | Yes, directly, in your own script |
| **ChatGPT** app, **Responses API** | Only through a bridge (below) |

**OpenAI Codex**: `~/.codex/config.toml` (TOML, not JSON):

```toml
[mcp_servers.tfbs]
command = "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe"
args = []
```

**Gemini CLI**: `~/.gemini/settings.json`, with the `mcpServers` shape shown above.

**OpenAI Agents SDK**: no config file; the script starts the server:

```python
import asyncio
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

TFBS = r"C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe"

async def main():
    async with MCPServerStdio(
        name="tfbs",
        params={"command": TFBS, "args": []},
        client_session_timeout_seconds=120,
    ) as server:
        agent = Agent(name="TFBS", instructions="Use the tfbs tools.",
                      mcp_servers=[server])
        result = await Runner.run(agent, "What environment is the TFBS server running in?")
        print(result.final_output)

asyncio.run(main())
```

### ChatGPT and the Responses API

These cannot start a program on your computer: OpenAI's servers make the call, so the server has to be reachable over HTTPS. A bridge turns it into a web service:

```bash
npx -y supergateway --stdio "C:/Users/you/tfbs-env/Scripts/tfbs-mcp.exe" --outputTransport streamableHttp --port 8000
```

That serves `http://localhost:8000/mcp`. ChatGPT can only reach it once it is exposed publicly (e.g. `cloudflared tunnel --url http://127.0.0.1:8000`) and the resulting HTTPS URL is added in ChatGPT's connector settings. Be careful with this: it puts a tool that reads files on your computer behind a public URL.

## Local and open-source models

| Runner | How |
|---|---|
| **LM Studio** | Sidebar: **Program > Install > Edit mcp.json**; `mcpServers` shape. Needs version 0.3.17 or later |
| **Jan** | **Settings > MCP Servers > Add Server**; put your path in Command |
| **LibreChat** | `mcpServers:` block in `librechat.yaml` |
| **Open WebUI** | Wrap with `mcpo` (below), then add the URL under Settings > Integrations > Tools |
| **Ollama** | No built-in MCP support; use a client in front of it, e.g. `ollmcp` |

Ollama needs a separate MCP client:

```bash
pip install --upgrade ollmcp
ollmcp --servers-json tfbs-servers.json --model qwen3:8b
```

Open WebUI needs the server turned into an OpenAPI service first:

```bash
uvx mcpo --port 8000 --api-key "choose-a-secret" -- "C:/Users/you/tfbs-env/Scripts/tfbs-mcp.exe"
```

Then add `http://localhost:8000` as a Tool in Open WebUI.
