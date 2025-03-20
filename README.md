# NSAF MCP Server

This is a Model Context Protocol (MCP) server for the Neuro-Symbolic Autonomy Framework (NSAF). It allows AI assistants to interact with the NSAF framework through the MCP protocol.

## Features

- Run NSAF evolution with customizable parameters
- Compare different NSAF agent architectures
- Integrate NSAF capabilities into AI assistants

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+ with the NSAF framework installed

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nsaf-mcp-server.git
cd nsaf-mcp-server
```

2. Install dependencies:
```bash
npm install
```

3. Build the server:
```bash
npm run build
```

## Configuration

The server requires the NSAF framework to be installed and accessible. You can configure the path to the NSAF project using the `NSAF_PROJECT_PATH` environment variable.

## Usage

### Running the server locally

```bash
npm start
```

### Deploying to GitHub

1. Create a new GitHub repository for your MCP server:
   - Go to GitHub and create a new repository named `nsaf-mcp-server`
   - Initialize it with a README file

2. Push your local code to the GitHub repository:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/nsaf-mcp-server.git
git push -u origin main
```

3. Configure GitHub Actions for CI/CD (optional):
   - Create a `.github/workflows` directory
   - Add a workflow file for testing and building the server

### Using with AI Assistants

To use this MCP server with AI assistants like Claude, you need to:

1. Install the server locally:
```bash
npm install -g nsaf-mcp-server
```

2. Add the server to your MCP settings configuration:

For Claude Desktop app, edit `~/Library/Application Support/Claude/claude_desktop_config.json` (on macOS):

```json
{
  "mcpServers": {
    "nsaf": {
      "command": "node",
      "args": ["/path/to/nsaf-mcp-server/build/index.js"],
      "env": {
        "NSAF_PROJECT_PATH": "/path/to/nsaf-project"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

For Cline, edit `/Users/onthego/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "nsaf": {
      "command": "node",
      "args": ["/path/to/nsaf-mcp-server/build/index.js"],
      "env": {
        "NSAF_PROJECT_PATH": "/path/to/nsaf-project"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

## Available Tools

### run_nsaf_evolution

Run NSAF evolution with specified parameters.

Parameters:
- `population_size`: Size of the agent population (default: 20)
- `generations`: Number of generations to evolve (default: 10)
- `mutation_rate`: Mutation rate (0.0-1.0) (default: 0.2)
- `crossover_rate`: Crossover rate (0.0-1.0) (default: 0.7)
- `architecture_complexity`: Complexity of the agent architecture ('simple', 'medium', 'complex') (default: 'medium')

### compare_nsaf_agents

Compare different NSAF agent architectures.

Parameters:
- `architectures`: List of architectures to compare (default: ['simple', 'medium', 'complex'])

## License

MIT
