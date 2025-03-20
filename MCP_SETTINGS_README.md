# MCP Settings Configuration Instructions

This file provides instructions for configuring the NSAF MCP server in your AI assistant's MCP settings.

## For Cline

1. Install the NSAF MCP server globally:
   ```bash
   npm install -g nsaf-mcp-server
   ```

2. Save the `cline_mcp_settings_example.json` file to:
   ```
   /Users/onthego/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
   ```

3. Restart Cline to load the new MCP server

## For Claude Desktop App

1. Install the NSAF MCP server globally:
   ```bash
   npm install -g nsaf-mcp-server
   ```

2. Edit the Claude Desktop configuration file:
   ```
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```

3. Add the NSAF MCP server configuration:
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

4. Restart the Claude Desktop app to load the new MCP server

## Troubleshooting

If the MCP server doesn't appear in your AI assistant:

1. Verify the NSAF MCP server is properly installed globally:
   ```bash
   which nsaf-mcp-server
   ```
   This should return a path to the installed binary.

2. Try reinstalling the package:
   ```bash
   npm uninstall -g nsaf-mcp-server
   npm install -g nsaf-mcp-server
   ```

3. Check the logs of your AI assistant for any error messages

4. If you're still having issues, you can try the direct path approach instead:
   - Find the path to the installed package: `npm list -g nsaf-mcp-server`
   - Update your configuration to use:
     ```json
     "command": "node",
     "args": ["/path/to/node_modules/nsaf-mcp-server/build/index.js"]
     ```
