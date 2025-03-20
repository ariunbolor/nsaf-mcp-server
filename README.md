# NSAF MCP Server

This is a Model Context Protocol (MCP) server for the Neuro-Symbolic Autonomy Framework (NSAF). It allows AI assistants to interact with the NSAF framework through the MCP protocol.

> **Note:** This repository includes both the NSAF framework code and the MCP server implementation, making it a complete package that can be deployed and used anywhere.

> **Note:** This implementation uses a simplified version of the MCP protocol that doesn't require the official MCP SDK. It implements the core functionality needed to expose NSAF capabilities to AI assistants.

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

The server includes the NSAF framework code, so no additional configuration is required for basic usage. The MCP server is designed to work out-of-the-box when installed globally.

## Usage

### Running the server locally

```bash
npm start
```

### Deploying to GitHub

1. Create a new GitHub repository for your MCP server:
   - Go to GitHub and create a new repository named `nsaf-mcp-server`
   - Initialize it with a README file

2. Use the provided setup script to push your code to GitHub:
```bash
# For a new repository
./setup-github-fixed.sh yourusername

# If the repository already exists and you want to overwrite its content
./setup-github-fixed.sh yourusername --force
```

The script will:
- Initialize git if needed
- Set up the remote repository
- Commit your changes
- Try to push to GitHub (with options to handle existing repositories)

3. Configure GitHub Actions for CI/CD (optional):
   - Create a `.github/workflows` directory
   - Add a workflow file for testing and building the server

### Using with AI Assistants

To use this MCP server with AI assistants like Claude, you need to:

1. Install the server:

   Option 1: Install from GitHub (after pushing your code):
   ```bash
   npm install -g yourusername/nsaf-mcp-server
   ```

   Option 2: Install from your local directory:
   ```bash
   # Navigate to the nsaf-mcp-server directory
   cd nsaf_mcp_server
   
   # Install dependencies and build
   npm install
   npm run build
   
   # Install globally from the local directory
   npm install -g .
   ```

2. Add the server to your MCP settings configuration:

For Claude Desktop app, edit `~/Library/Application Support/Claude/claude_desktop_config.json` (on macOS):

```json
{
  "mcpServers": {
    "nsaf": {
      "command": "nsaf-mcp-server",
      "args": [],
      "env": {},
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
      "command": "nsaf-mcp-server",
      "args": [],
      "env": {},
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
