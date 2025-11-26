#!/usr/bin/env python3
"""
MCP Interface Test
==================

Test script for NSAF MCP interface functionality.
"""

import asyncio
import json
from core.mcp_interface import NSAFMCPServer


async def test_mcp_interface():
    """Test MCP interface functionality."""
    print("🔌 Testing NSAF MCP Interface...")
    
    try:
        # Create MCP server
        server = NSAFMCPServer()
        
        # Test 1: Get status
        print("\n1. Testing get_nsaf_status...")
        result = await server.handle_tool_call('get_nsaf_status', {})
        print(f"   Result: {result.get('success', False)}")
        if result.get('success'):
            status = result['result']
            print(f"   Active agents: {status.get('active_agents', 0)}")
            print(f"   Task clusters: {status.get('task_clusters', 0)}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Test 2: Task clustering
        print("\n2. Testing cluster_nsaf_tasks...")
        cluster_args = {
            'problem_description': 'Design a web application with user authentication',
            'max_clusters': 5
        }
        result = await server.handle_tool_call('cluster_nsaf_tasks', cluster_args)
        print(f"   Result: {result.get('success', False)}")
        if result.get('success'):
            clusters = result['result'].get('clusters_created', 0)
            print(f"   Clusters created: {clusters}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Test 3: Intent projection
        print("\n3. Testing project_nsaf_intent...")
        intent_args = {
            'intent_description': 'Build a machine learning model for data analysis',
            'projection_depth': 2
        }
        result = await server.handle_tool_call('project_nsaf_intent', intent_args)
        print(f"   Result: {result.get('success', False)}")
        if result.get('success'):
            projections = result['result'].get('projections', [])
            print(f"   Projections: {len(projections)}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print("\n✅ MCP Interface tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ MCP Interface test failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(test_mcp_interface())