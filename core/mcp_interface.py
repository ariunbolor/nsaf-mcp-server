"""
Model Context Protocol (MCP) Interface for NSAF
===============================================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Provides MCP protocol support for the Neuro-Symbolic Autonomy Framework,
enabling integration with AI assistants like Claude.
"""

import json
import sys
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .framework import NeuroSymbolicAutonomyFramework


@dataclass
class MCPTool:
    """Represents an MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class NSAFMCPServer:
    """
    MCP Server for the Neuro-Symbolic Autonomy Framework.
    
    Exposes NSAF capabilities through the Model Context Protocol,
    allowing AI assistants to interact with the full framework.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the MCP server with NSAF framework.
        
        Args:
            config_path: Path to NSAF configuration file
        """
        self.framework = None
        self.config_path = config_path
        self.tools = self._define_tools()
        
    def _define_tools(self) -> List[MCPTool]:
        """Define available MCP tools for NSAF operations."""
        return [
            MCPTool(
                name="run_nsaf_evolution",
                description="Run NSAF evolution with specified parameters to create optimized agents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the task for agent optimization"
                        },
                        "population_size": {
                            "type": "integer",
                            "description": "Size of the agent population",
                            "default": 20,
                            "minimum": 5,
                            "maximum": 100
                        },
                        "generations": {
                            "type": "integer", 
                            "description": "Number of generations to evolve",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        },
                        "architecture_complexity": {
                            "type": "string",
                            "enum": ["simple", "medium", "complex"],
                            "description": "Complexity of agent architecture",
                            "default": "medium"
                        },
                        "goals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "target": {"type": "number"},
                                    "priority": {"type": "number"}
                                }
                            },
                            "description": "Performance goals for the agents"
                        }
                    },
                    "required": ["task_description"]
                }
            ),
            MCPTool(
                name="analyze_nsaf_memory",
                description="Analyze the symbolic memory graph and retrieve insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search the memory graph"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="project_nsaf_intent",
                description="Project future intents and planning using recursive intent projection",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "intent_description": {
                            "type": "string",
                            "description": "Description of the intent to project"
                        },
                        "projection_depth": {
                            "type": "integer",
                            "description": "Depth of recursive projection",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "description": "Minimum confidence threshold for projections",
                            "default": 0.7,
                            "minimum": 0.1,
                            "maximum": 1.0
                        }
                    },
                    "required": ["intent_description"]
                }
            ),
            MCPTool(
                name="cluster_nsaf_tasks",
                description="Use quantum-symbolic task clustering to decompose complex problems",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem_description": {
                            "type": "string",
                            "description": "Description of the complex problem to decompose"
                        },
                        "max_clusters": {
                            "type": "integer", 
                            "description": "Maximum number of task clusters",
                            "default": 10
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Similarity threshold for clustering",
                            "default": 0.85
                        }
                    },
                    "required": ["problem_description"]
                }
            ),
            MCPTool(
                name="get_nsaf_status",
                description="Get current status of the NSAF framework",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            )
        ]
    
    async def initialize_framework(self):
        """Initialize the NSAF framework lazily."""
        if self.framework is None:
            self.framework = NeuroSymbolicAutonomyFramework(self.config_path)
            
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP tool calls.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool execution
        """
        await self.initialize_framework()
        
        try:
            if tool_name == "run_nsaf_evolution":
                return await self._run_evolution(arguments)
            elif tool_name == "analyze_nsaf_memory":
                return await self._analyze_memory(arguments)
            elif tool_name == "project_nsaf_intent":
                return await self._project_intent(arguments)
            elif tool_name == "cluster_nsaf_tasks":
                return await self._cluster_tasks(arguments)
            elif tool_name == "get_nsaf_status":
                return await self._get_status(arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name
            }
    
    async def _run_evolution(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run NSAF evolution process."""
        task = {
            "description": args["task_description"],
            "goals": args.get("goals", []),
            "constraints": [{
                "type": "complexity",
                "value": args.get("architecture_complexity", "medium")
            }],
            "requirements": {
                "population_size": args.get("population_size", 20),
                "generations": args.get("generations", 10)
            }
        }
        
        result = await self.framework.process_task(task)
        
        return {
            "success": True,
            "result": {
                "task_clusters": len(result.get("task_clusters", [])),
                "agents_created": len(result.get("agents", [])),
                "evolution_metrics": result.get("metrics", {}),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _analyze_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the NSAF memory graph."""
        query = args["query"]
        max_results = args.get("max_results", 10)
        
        # Search memory graph
        memories = self.framework.memory_tuning.search_memories(
            query_symbols=query.split(),
            max_results=max_results
        )
        
        return {
            "success": True,
            "result": {
                "query": query,
                "memories_found": len(memories),
                "memories": [
                    {
                        "id": mem["id"],
                        "content": mem["content"],
                        "confidence": mem["confidence"],
                        "timestamp": mem.get("timestamp")
                    }
                    for mem in memories[:max_results]
                ]
            }
        }
    
    async def _project_intent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Project future intents using RIP."""
        from .recursive_intent import Intent
        
        intent = Intent(
            id=f"mcp_intent_{int(datetime.now().timestamp())}",
            description=args["intent_description"],
            goals=[],
            constraints=[],
            context={"source": "mcp"},
            confidence=args.get("confidence_threshold", 0.7),
            timestamp=datetime.now().timestamp()
        )
        
        # Create a basic current state
        current_state = {
            "depth": args.get("projection_depth", 3),
            "confidence_threshold": args.get("confidence_threshold", 0.7),
            "resources": {"available": True}
        }
        
        projections = self.framework.recursive_intent.project_intent(
            intent,
            current_state
        )
        
        return {
            "success": True,
            "result": {
                "intent": args["intent_description"],
                "projections": [
                    {
                        "step": i + 1,
                        "description": proj.intent.description,
                        "confidence": proj.intent.confidence,
                        "success_probability": proj.success_probability,
                        "required_capabilities": proj.required_capabilities,
                        "adaptation_steps": len(proj.adaptation_steps)
                    }
                    for i, proj in enumerate(projections)
                ]
            }
        }
    
    async def _cluster_tasks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster tasks using quantum-symbolic clustering."""
        problem = {
            "description": args["problem_description"],
            "tasks": [{"description": args["problem_description"]}]
        }
        
        clusters = self.framework.task_clustering.decompose_problem(problem)
        
        return {
            "success": True,
            "result": {
                "problem": args["problem_description"],
                "clusters_created": len(clusters),
                "clusters": [
                    {
                        "id": cluster.id,
                        "tasks": len(cluster.tasks),
                        "similarity_score": cluster.similarity_score
                    }
                    for cluster in clusters
                ]
            }
        }
    
    async def _get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get NSAF framework status."""
        status = self.framework.get_system_status()
        agents = self.framework.get_active_agents()
        clusters = self.framework.get_task_clusters()
        
        return {
            "success": True,
            "result": {
                "framework_status": status,
                "active_agents": len(agents),
                "task_clusters": len(clusters),
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "task_clustering": "active",
                    "scma": "active", 
                    "memory_tuning": "active",
                    "recursive_intent": "active",
                    "human_ai_synergy": "active"
                }
            }
        }


async def main():
    """Main MCP server entry point."""
    server = NSAFMCPServer()
    
    # Simple MCP protocol implementation
    while True:
        try:
            line = input()
            request = json.loads(line)
            
            if request.get("method") == "tools/list":
                response = {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in server.tools
                    ]
                }
            elif request.get("method") == "tools/call":
                tool_name = request["params"]["name"]
                arguments = request["params"].get("arguments", {})
                response = await server.handle_tool_call(tool_name, arguments)
            else:
                response = {"error": "Unknown method"}
                
            print(json.dumps(response))
            
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            error_response = {"error": f"Server error: {str(e)}"}
            print(json.dumps(error_response))


if __name__ == "__main__":
    asyncio.run(main())