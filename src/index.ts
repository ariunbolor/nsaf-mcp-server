#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'child_process';
import { join } from 'path';
import * as fs from 'fs';

// Path to the NSAF Python project
const NSAF_PROJECT_PATH = process.env.NSAF_PROJECT_PATH || '../';

// Check if the NSAF project exists
if (!fs.existsSync(join(NSAF_PROJECT_PATH, 'main.py'))) {
  throw new Error(`NSAF project not found at ${NSAF_PROJECT_PATH}. Please set the NSAF_PROJECT_PATH environment variable.`);
}

/**
 * Execute a Python script from the NSAF project
 */
async function executePythonScript(scriptPath: string, args: string[] = []): Promise<string> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath, ...args], {
      cwd: NSAF_PROJECT_PATH,
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}: ${stderr}`));
      } else {
        resolve(stdout);
      }
    });
  });
}

/**
 * NSAF MCP Server
 */
class NSAFServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'nsaf-mcp-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'run_nsaf_evolution',
          description: 'Run NSAF evolution with specified parameters',
          inputSchema: {
            type: 'object',
            properties: {
              population_size: {
                type: 'number',
                description: 'Size of the agent population',
                default: 20,
              },
              generations: {
                type: 'number',
                description: 'Number of generations to evolve',
                default: 10,
              },
              mutation_rate: {
                type: 'number',
                description: 'Mutation rate (0.0-1.0)',
                default: 0.2,
              },
              crossover_rate: {
                type: 'number',
                description: 'Crossover rate (0.0-1.0)',
                default: 0.7,
              },
              architecture_complexity: {
                type: 'string',
                description: 'Complexity of the agent architecture',
                enum: ['simple', 'medium', 'complex'],
                default: 'medium',
              },
            },
            required: [],
          },
        },
        {
          name: 'compare_nsaf_agents',
          description: 'Compare different NSAF agent architectures',
          inputSchema: {
            type: 'object',
            properties: {
              architectures: {
                type: 'array',
                description: 'List of architectures to compare',
                items: {
                  type: 'string',
                  enum: ['simple', 'medium', 'complex'],
                },
                default: ['simple', 'medium', 'complex'],
              },
            },
            required: [],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      switch (request.params.name) {
        case 'run_nsaf_evolution':
          return this.handleRunNSAFEvolution(request.params.arguments);
        case 'compare_nsaf_agents':
          return this.handleCompareNSAFAgents(request.params.arguments);
        default:
          throw new McpError(
            ErrorCode.MethodNotFound,
            `Unknown tool: ${request.params.name}`
          );
      }
    });
  }

  private async handleRunNSAFEvolution(args: any): Promise<any> {
    try {
      // Set environment variables for the NSAF script
      process.env.NSAF_POPULATION_SIZE = args?.population_size?.toString() || '20';
      process.env.NSAF_GENERATIONS = args?.generations?.toString() || '10';
      process.env.NSAF_MUTATION_RATE = args?.mutation_rate?.toString() || '0.2';
      process.env.NSAF_CROSSOVER_RATE = args?.crossover_rate?.toString() || '0.7';
      process.env.NSAF_ARCHITECTURE_COMPLEXITY = args?.architecture_complexity || 'medium';

      // Execute the main.py script
      const result = await executePythonScript('main.py');

      return {
        content: [
          {
            type: 'text',
            text: result,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error running NSAF evolution: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleCompareNSAFAgents(args: any): Promise<any> {
    try {
      // Set environment variables for the NSAF script
      process.env.NSAF_ARCHITECTURES = JSON.stringify(args?.architectures || ['simple', 'medium', 'complex']);

      // Execute a custom script to compare agents
      // Note: This would require creating a custom script in the NSAF project
      const result = await executePythonScript('run_example.py');

      return {
        content: [
          {
            type: 'text',
            text: result,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error comparing NSAF agents: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('NSAF MCP server running on stdio');
  }
}

const server = new NSAFServer();
server.run().catch(console.error);
