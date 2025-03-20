# MCP Settings Configuration Instructions

This file provides instructions for configuring the NSAF MCP server in your AI assistant's MCP settings.

## For Cline

1. Save the `cline_mcp_settings_example.json` file to:
   ```
   /Users/onthego/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
   ```

2. Edit the file to update the paths:
   - Replace `/path/to/nsaf-mcp-server/build/index.js` with the actual path to the built index.js file
     - If installed globally: Use the output of `which nsaf-mcp-server` in your terminal
     - If installed locally: `/Users/onthego/Documents/agent/nsaf_mcp_server/build/index.js`
   
   - Replace `/path/to/nsaf-project` with the actual path to your NSAF project
     - Example: `/Users/onthego/Documents/agent`

3. Restart Cline to load the new MCP server

## For Claude Desktop App

1. Edit the Claude Desktop configuration file:
   ```
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```

2. Add the NSAF MCP server configuration:
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

3. Update the paths as described in the Cline instructions above

4. Restart the Claude Desktop app to load the new MCP server

## Troubleshooting

If the MCP server doesn't appear in your AI assistant:

1. Check that the paths in the configuration file are correct
2. Ensure the NSAF MCP server is properly installed and built
3. Verify that the NSAF project path exists and contains the required files
4. Check the logs of your AI assistant for any error messages
