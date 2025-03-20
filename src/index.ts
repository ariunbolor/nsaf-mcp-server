#!/usr/bin/env node
import { spawn } from 'child_process';
import { join } from 'path';
import * as fs from 'fs';
import * as readline from 'readline';

// Path to the NSAF Python project - now included in the same repository
const NSAF_PROJECT_PATH = process.env.NSAF_PROJECT_PATH || './';

// Check if the NSAF project exists
if (!fs.existsSync(join(NSAF_PROJECT_PATH, 'main.py'))) {
  console.error(`Warning: NSAF main.py not found at ${NSAF_PROJECT_PATH}. Using default path.`);
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
 * Simplified MCP Server for NSAF
 */
class SimplifiedMCPServer {
  private rl: readline.Interface;

  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    process.on('SIGINT', () => {
      this.rl.close();
      process.exit(0);
    });
  }

  async run() {
    console.error('NSAF MCP server running...');
    
    this.rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        const response = await this.handleRequest(request);
        console.log(JSON.stringify(response));
      } catch (error) {
        console.error('Error handling request:', error);
        console.log(JSON.stringify({
          jsonrpc: '2.0',
          id: null,
          error: {
            code: -32603,
            message: error instanceof Error ? error.message : String(error)
          }
        }));
      }
    });
  }

  private async handleRequest(request: any): Promise<any> {
    if (!request.method) {
      return {
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: -32601,
          message: 'Method not found'
        }
      };
    }

    switch (request.method) {
      case 'list_tools':
        return this.handleListTools(request);
      case 'call_tool':
        return this.handleCallTool(request);
      default:
        return {
          jsonrpc: '2.0',
          id: request.id,
          error: {
            code: -32601,
            message: `Method not found: ${request.method}`
          }
        };
    }
  }

  private async handleListTools(request: any): Promise<any> {
    return {
      jsonrpc: '2.0',
      id: request.id,
      result: {
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
        ]
      }
    };
  }

  private async handleCallTool(request: any): Promise<any> {
    const { name, arguments: args } = request.params;

    if (!name) {
      return {
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: -32602,
          message: 'Invalid params: missing tool name'
        }
      };
    }

    try {
      let result;
      switch (name) {
        case 'run_nsaf_evolution':
          result = await this.handleRunNSAFEvolution(args || {});
          break;
        case 'compare_nsaf_agents':
          result = await this.handleCompareNSAFAgents(args || {});
          break;
        default:
          return {
            jsonrpc: '2.0',
            id: request.id,
            error: {
              code: -32601,
              message: `Tool not found: ${name}`
            }
          };
      }

      return {
        jsonrpc: '2.0',
        id: request.id,
        result
      };
    } catch (error) {
      return {
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: -32603,
          message: error instanceof Error ? error.message : String(error)
        }
      };
    }
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
      throw new Error(`Error running NSAF evolution: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private async handleCompareNSAFAgents(args: any): Promise<any> {
    try {
      // Set environment variables for the NSAF script
      process.env.NSAF_ARCHITECTURES = JSON.stringify(args?.architectures || ['simple', 'medium', 'complex']);

      // Execute a custom script to compare agents
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
      throw new Error(`Error comparing NSAF agents: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

const server = new SimplifiedMCPServer();
server.run().catch(console.error);
