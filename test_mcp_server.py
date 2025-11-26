#!/usr/bin/env python3
"""
NSAF MCP Server Test Suite
===========================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Comprehensive test suite for the NSAF MCP Server.
"""

import asyncio
import json
from nsaf_mcp_server import NSAFMCPServer


async def test_framework_management():
    """Test framework management tools."""
    print("🔧 Testing Framework Management")
    print("-" * 40)
    
    server = NSAFMCPServer()
    
    # Test initialization
    print("1. Testing framework initialization...")
    result = await server.handle_tool_call("initialize_nsaf_framework", {
        "config_path": "config/config.yaml"
    })
    print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
    if result.get('success'):
        status = result['result']['status']
        print(f"   Framework Status: {status}")
    
    # Test status check
    print("\n2. Testing status retrieval...")
    result = await server.handle_tool_call("get_nsaf_status", {
        "include_details": True
    })
    print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
    if result.get('success'):
        metrics = result['result']['metrics']
        print(f"   Active Agents: {metrics['active_agents']}")
        print(f"   Task Clusters: {metrics['task_clusters']}")
        print(f"   Active Tasks: {metrics['active_tasks']}")
    
    # Test shutdown
    print("\n3. Testing framework shutdown...")
    result = await server.handle_tool_call("shutdown_nsaf_framework", {})
    print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
    
    return True


async def test_task_processing():
    """Test task processing capabilities."""
    print("\n🧠 Testing Task Processing")
    print("-" * 40)
    
    server = NSAFMCPServer()
    
    # Initialize framework first
    await server.handle_tool_call("initialize_nsaf_framework", {})
    
    # Test complex task processing
    print("1. Testing complex task processing...")
    complex_task = {
        "description": "Design an intelligent recommendation system",
        "goals": [
            {"type": "accuracy", "target": 0.92, "priority": 1.0, "complexity": 0.8, "progress": 0.0},
            {"type": "latency", "target": 100, "priority": 0.8, "complexity": 0.6, "progress": 0.0}
        ],
        "constraints": [
            {"type": "memory", "limit": "8GB", "importance": 0.9, "strictness": 0.7}
        ],
        "requirements": {
            "frameworks": ["pytorch", "scikit-learn"],
            "data_sources": ["user_behavior", "item_features"],
            "deployment": "cloud"
        },
        "tasks": [
            {"name": "data_preprocessing", "type": "pipeline", "priority": 1.0, "dependencies": []},
            {"name": "model_training", "type": "ml", "priority": 0.9, "dependencies": ["data_preprocessing"]},
            {"name": "deployment", "type": "deployment", "priority": 0.7, "dependencies": ["model_training"]}
        ],
        "complexity": 0.8
    }
    
    result = await server.handle_tool_call("process_complex_task", complex_task)
    print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
    
    task_id = None
    if result.get('success'):
        task_id = result['result']['task_id']
        summary = result['result']['processing_summary']
        print(f"   Task ID: {task_id}")
        print(f"   Task Clusters: {summary['task_clusters']}")
        print(f"   Agents Created: {summary['agents_created']}")
        print(f"   Status: {result['result']['status']}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Test task status retrieval
    if task_id:
        print(f"\n2. Testing task status retrieval...")
        result = await server.handle_tool_call("get_task_status", {
            "task_id": task_id
        })
        print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
        if result.get('success'):
            task_status = result['result']
            print(f"   Task Status: {task_status['status']}")
            print(f"   Progress: {task_status['progress']}")
    
    await server.handle_tool_call("shutdown_nsaf_framework", {})
    return True


async def test_component_tools():
    """Test individual component tools."""
    print("\n⚛️ Testing Component Tools")
    print("-" * 40)
    
    server = NSAFMCPServer()
    await server.handle_tool_call("initialize_nsaf_framework", {})
    
    # Test quantum clustering
    print("1. Testing quantum task clustering...")
    result = await server.handle_tool_call("cluster_tasks_quantum", {
        "problem_description": "Optimize machine learning pipeline",
        "tasks": [
            {"name": "data_collection", "description": "Gather training data"},
            {"name": "preprocessing", "description": "Clean and prepare data"},
            {"name": "training", "description": "Train ML models"},
            {"name": "evaluation", "description": "Evaluate model performance"}
        ],
        "max_clusters": 3
    })
    print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
    if result.get('success'):
        clusters = result['result']['clusters_created']
        print(f"   Clusters Created: {clusters}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Test intent projection
    print("\n2. Testing recursive intent projection...")
    result = await server.handle_tool_call("project_intent_recursive", {
        "intent_description": "Create an automated code review system",
        "goals": [
            {"type": "accuracy", "target": 0.9, "priority": 0.8}
        ],
        "projection_depth": 3,
        "confidence_threshold": 0.6
    })
    print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
    if result.get('success'):
        projections = result['result']['projections_generated']
        print(f"   Projections Generated: {projections}")
        if result['result']['projections']:
            first_proj = result['result']['projections'][0]
            print(f"   First Projection Confidence: {first_proj['confidence']}")
    
    # Test memory operations
    print("\n3. Testing memory system...")
    
    # Add memory
    add_result = await server.handle_tool_call("add_memory", {
        "content": "Machine learning optimization techniques are crucial for performance",
        "memory_type": "knowledge",
        "importance": 0.8
    })
    print(f"   Add Memory: {'✅ SUCCESS' if add_result.get('success') else '❌ FAILED'}")
    
    # Query memory
    query_result = await server.handle_tool_call("query_memory", {
        "query": "machine learning optimization",
        "max_results": 5
    })
    print(f"   Query Memory: {'✅ SUCCESS' if query_result.get('success') else '❌ FAILED'}")
    if query_result.get('success'):
        memories = query_result['result']['results_found']
        print(f"   Memories Found: {memories}")
    
    # Get memory metrics
    metrics_result = await server.handle_tool_call("get_memory_metrics", {})
    print(f"   Memory Metrics: {'✅ SUCCESS' if metrics_result.get('success') else '❌ FAILED'}")
    if metrics_result.get('success'):
        metrics = metrics_result['result']['metrics']
        print(f"   Total Memories: {metrics['total_memories']}")
        print(f"   Connections: {metrics['total_connections']}")
    
    # Test agent evolution
    print("\n4. Testing agent evolution...")
    result = await server.handle_tool_call("evolve_agents_scma", {
        "task_description": "Optimize neural network architectures",
        "population_size": 10,
        "generations": 5,
        "architecture_complexity": "medium"
    })
    print(f"   Status: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
    if result.get('success'):
        agents = result['result']['agents_created']
        fitness = result['result']['best_fitness']
        print(f"   Agents Created: {agents}")
        print(f"   Best Fitness: {fitness}")
    
    await server.handle_tool_call("shutdown_nsaf_framework", {})
    return True


async def test_tool_discovery():
    """Test MCP tool discovery."""
    print("\n🔍 Testing Tool Discovery")
    print("-" * 40)
    
    server = NSAFMCPServer()
    
    print(f"Total Tools Available: {len(server.tools)}")
    print("\nTool Categories:")
    
    categories = {
        "Framework Management": ["initialize_nsaf_framework", "get_nsaf_status", "shutdown_nsaf_framework"],
        "Task Processing": ["process_complex_task", "get_task_status", "update_task_state"],
        "Quantum-Symbolic": ["cluster_tasks_quantum"],
        "Meta-Agents": ["evolve_agents_scma", "get_active_agents"], 
        "Memory System": ["add_memory", "query_memory", "get_memory_metrics"],
        "Intent Projection": ["project_intent_recursive"],
        "Human-AI Synergy": ["synchronize_cognitive_state"],
        "Foundation Models": ["generate_with_foundation_models", "get_embeddings"],
        "System Analytics": ["analyze_system_performance", "update_configuration", "get_configuration"]
    }
    
    for category, tools in categories.items():
        available_tools = [tool.name for tool in server.tools if tool.name in tools]
        print(f"   {category}: {len(available_tools)}/{len(tools)} tools")
        for tool in available_tools:
            print(f"      ✅ {tool}")
    
    return True


async def demonstrate_mcp_capabilities():
    """Demonstrate key MCP capabilities with practical examples."""
    print("\n🚀 MCP Capability Demonstration")
    print("=" * 50)
    
    # Example 1: End-to-end task processing
    print("\n📋 Example 1: E2E Task Processing")
    print("   Task: Build AI-powered customer service system")
    
    server = NSAFMCPServer()
    await server.handle_tool_call("initialize_nsaf_framework", {})
    
    task_result = await server.handle_tool_call("process_complex_task", {
        "description": "Build AI-powered customer service system",
        "goals": [
            {"type": "response_accuracy", "target": 0.95, "priority": 1.0, "complexity": 0.8, "progress": 0.0},
            {"type": "response_time", "target": 2.0, "priority": 0.9, "complexity": 0.6, "progress": 0.0},
            {"type": "customer_satisfaction", "target": 0.9, "priority": 0.8, "complexity": 0.7, "progress": 0.0}
        ],
        "constraints": [
            {"type": "budget", "limit": 100000, "importance": 0.9, "strictness": 0.8},
            {"type": "timeline", "limit": "3 months", "importance": 0.8, "strictness": 0.7}
        ],
        "tasks": [
            {"name": "intent_classification", "type": "ml", "priority": 1.0, "dependencies": []},
            {"name": "response_generation", "type": "nlp", "priority": 0.9, "dependencies": ["intent_classification"]},
            {"name": "sentiment_analysis", "type": "ml", "priority": 0.8, "dependencies": []},
            {"name": "escalation_detection", "type": "ml", "priority": 0.7, "dependencies": ["sentiment_analysis"]},
            {"name": "integration_layer", "type": "api", "priority": 0.6, "dependencies": ["response_generation", "escalation_detection"]}
        ],
        "complexity": 0.85
    })
    
    if task_result.get('success'):
        print(f"   ✅ Task processed successfully")
        print(f"   📊 Summary: {task_result['result']['processing_summary']}")
    else:
        print(f"   ❌ Task processing failed: {task_result.get('error')}")
    
    # Example 2: Intent projection for planning
    print("\n🔮 Example 2: Strategic Planning with Intent Projection")
    print("   Intent: Develop autonomous research capabilities")
    
    intent_result = await server.handle_tool_call("project_intent_recursive", {
        "intent_description": "Develop autonomous research capabilities for scientific discovery",
        "goals": [
            {"type": "research_quality", "target": 0.9, "priority": 1.0},
            {"type": "automation_level", "target": 0.8, "priority": 0.9}
        ],
        "projection_depth": 5,
        "confidence_threshold": 0.7
    })
    
    if intent_result.get('success'):
        projections = intent_result['result']['projections']
        print(f"   ✅ Generated {len(projections)} projection steps")
        for i, proj in enumerate(projections[:3]):  # Show first 3
            print(f"      Step {i+1}: Confidence {proj['confidence']:.2f}")
    
    # Example 3: Adaptive agent creation
    print("\n🤖 Example 3: Adaptive Agent Evolution")
    print("   Goal: Create specialized code analysis agents")
    
    agent_result = await server.handle_tool_call("evolve_agents_scma", {
        "task_description": "Automated code quality analysis and improvement",
        "population_size": 15,
        "generations": 8,
        "fitness_criteria": [
            {"name": "code_quality_detection", "weight": 0.4, "target": 0.9},
            {"name": "performance_optimization", "weight": 0.3, "target": 0.8},
            {"name": "security_analysis", "weight": 0.3, "target": 0.85}
        ],
        "architecture_complexity": "complex"
    })
    
    if agent_result.get('success'):
        agents = agent_result['result']['agents_created'] 
        fitness = agent_result['result']['best_fitness']
        print(f"   ✅ Evolved {agents} specialized agents")
        print(f"   🏆 Best fitness achieved: {fitness}")
    
    await server.handle_tool_call("shutdown_nsaf_framework", {})
    
    print(f"\n🎯 MCP Server Capability Summary:")
    print(f"   • Complete NSAF framework exposed as {len(server.tools)} MCP tools")
    print(f"   • Multi-objective task processing with quantum enhancement")
    print(f"   • Neural intent projection for strategic planning") 
    print(f"   • Evolutionary agent optimization for specialized tasks")
    print(f"   • Symbolic memory system for knowledge management")
    print(f"   • Human-AI synergy for collaborative intelligence")
    print(f"   • Foundation model integration across providers")
    print(f"   • Real-time monitoring and system analytics")


async def main():
    """Run comprehensive MCP server test suite."""
    print("🧪 NSAF MCP Server - Comprehensive Test Suite")
    print("=" * 60)
    print("Author: Bolorerdene Bundgaa (https://bolor.me)")
    print()
    
    try:
        # Run test suites
        await test_tool_discovery()
        await test_framework_management()
        await test_task_processing() 
        await test_component_tools()
        await demonstrate_mcp_capabilities()
        
        print(f"\n🎉 All MCP Server Tests Completed Successfully!")
        print(f"\n📚 Usage Instructions:")
        print(f"   1. Start MCP server: python3 nsaf_mcp_server.py")
        print(f"   2. Connect AI assistant via MCP protocol")
        print(f"   3. Use any of the {len(NSAFMCPServer().tools)} available tools")
        print(f"   4. Access full NSAF capabilities through natural language")
        
        print(f"\n🔧 Available Tool Categories:")
        print(f"   • Framework Management (3 tools)")
        print(f"   • Task Processing (3 tools)")
        print(f"   • Quantum-Symbolic Computing (1 tool)")
        print(f"   • Meta-Agent Evolution (2 tools)")
        print(f"   • Memory Management (3 tools)")
        print(f"   • Intent Projection (1 tool)")
        print(f"   • Human-AI Synergy (1 tool)")
        print(f"   • Foundation Models (2 tools)")
        print(f"   • System Analytics (3 tools)")
        
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())