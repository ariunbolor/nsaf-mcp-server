#!/usr/bin/env python3
"""
Unified NSAF Framework Example
=============================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Demonstrates the complete, integrated Neuro-Symbolic Autonomy Framework
with all components working together.
"""

import asyncio
import json
import os
from pathlib import Path
from core import NeuroSymbolicAutonomyFramework, NSAFMCPServer


async def main():
    """
    Main demonstration of the unified NSAF framework.
    """
    print("🚀 Neuro-Symbolic Autonomy Framework (NSAF) - Unified Demo")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Initialize the complete NSAF framework
    print("\n🔧 Initializing NSAF Framework...")
    framework = NeuroSymbolicAutonomyFramework()
    
    # Define a comprehensive task
    task = {
        'description': 'Develop an AI system for predictive maintenance of industrial equipment',
        'goals': [
            {
                'type': 'accuracy',
                'target': 0.95,
                'priority': 0.9
            },
            {
                'type': 'latency',
                'target': 50,  # milliseconds
                'priority': 0.8
            },
            {
                'type': 'reliability',
                'target': 0.99,
                'priority': 0.85
            }
        ],
        'constraints': [
            {
                'type': 'memory',
                'limit': '8GB',
                'importance': 0.9
            },
            {
                'type': 'runtime',
                'limit': '12h',
                'importance': 0.7
            },
            {
                'type': 'cost',
                'limit': 1000,  # USD
                'importance': 0.6
            }
        ],
        'requirements': {
            'frameworks': ['pytorch', 'scikit-learn', 'pandas'],
            'data_sources': ['sensor_data', 'maintenance_logs', 'failure_reports'],
            'deployment': 'edge_device',
            'scalability': 'high',
            'explainability': 'required'
        },
        'tasks': [
            {
                'name': 'data_ingestion',
                'type': 'pipeline',
                'priority': 1.0,
                'dependencies': []
            },
            {
                'name': 'feature_engineering',
                'type': 'analysis',
                'priority': 0.95,
                'dependencies': ['data_ingestion']
            },
            {
                'name': 'anomaly_detection',
                'type': 'ml',
                'priority': 0.9,
                'dependencies': ['feature_engineering']
            },
            {
                'name': 'predictive_modeling',
                'type': 'ml',
                'priority': 1.0,
                'dependencies': ['anomaly_detection']
            },
            {
                'name': 'model_interpretation',
                'type': 'analysis',
                'priority': 0.8,
                'dependencies': ['predictive_modeling']
            },
            {
                'name': 'deployment_optimization',
                'type': 'deployment',
                'priority': 0.7,
                'dependencies': ['model_interpretation']
            }
        ],
        'complexity': 0.85
    }
    
    try:
        # Process task through the complete NSAF pipeline
        print("\n🧠 Processing task through NSAF pipeline...")
        print(f"Task: {task['description']}")
        print(f"Goals: {len(task['goals'])} objectives")
        print(f"Constraints: {len(task['constraints'])} limitations")
        print(f"Subtasks: {len(task['tasks'])} components")
        
        # Execute the full framework
        result = await framework.process_task(task)
        
        print("\n✅ Task Processing Complete!")
        print("\n📊 Results Summary:")
        print(f"  • Task clusters created: {len(result.get('task_clusters', []))}")
        print(f"  • Agents generated: {len(result.get('agents', []))}")
        print(f"  • Memory nodes: {len(result.get('memory_nodes', []))}")
        print(f"  • Intent projections: {len(result.get('intent_projections', []))}")
        
        # Display detailed results
        if result.get('task_clusters'):
            print(f"\n🔍 Task Clustering Results:")
            for i, cluster in enumerate(result['task_clusters'][:3]):  # Show first 3
                print(f"  Cluster {i+1}: {cluster.get('id', 'Unknown')} - {len(cluster.get('tasks', []))} tasks")
        
        if result.get('agents'):
            print(f"\n🤖 Generated Agents:")
            for i, agent in enumerate(result['agents'][:3]):  # Show first 3
                fitness = agent.get('fitness', 0)
                print(f"  Agent {i+1}: Fitness {fitness:.3f}")
        
        # Test state update
        print("\n📈 Updating task state...")
        if result.get('task_clusters'):
            cluster_id = result['task_clusters'][0].get('id')
            if cluster_id:
                state_update = {
                    'progress': 0.75,
                    'metrics': {
                        'accuracy': 0.92,
                        'latency': 45,
                        'reliability': 0.98
                    },
                    'resource_usage': {
                        'memory': '5.2GB',
                        'cpu': 0.65,
                        'gpu': 0.40
                    },
                    'status': 'optimization',
                    'understanding': 0.88,
                    'confidence': 0.85
                }
                
                update_result = await framework.update_task_state(cluster_id, state_update)
                print("  ✅ State update successful")
        
        # System status
        print("\n🔍 System Status:")
        status = framework.get_system_status()
        print(f"  • Framework version: {status.get('version', 'Unknown')}")
        print(f"  • Active components: {len(status.get('components', {}))}")
        print(f"  • Memory usage: {status.get('memory_usage', 'Unknown')}")
        
        # Active agents
        agents = framework.get_active_agents()
        print(f"  • Active agents: {len(agents)}")
        
        # Task clusters
        clusters = framework.get_task_clusters()
        print(f"  • Task clusters: {len(clusters)}")
        
        # Demonstrate MCP interface
        print("\n🔌 Testing MCP Interface...")
        mcp_server = NSAFMCPServer()
        
        # Test MCP tool call
        mcp_result = await mcp_server.handle_tool_call(
            "get_nsaf_status",
            {}
        )
        
        if mcp_result.get('success'):
            mcp_status = mcp_result['result']
            print(f"  ✅ MCP Interface working")
            print(f"  • Active agents: {mcp_status.get('active_agents', 0)}")
            print(f"  • Task clusters: {mcp_status.get('task_clusters', 0)}")
        else:
            print(f"  ❌ MCP Interface error: {mcp_result.get('error', 'Unknown')}")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Check for common issues
        if "qiskit" in str(e).lower():
            print("\n💡 Tip: Install quantum computing dependencies:")
            print("   pip install qiskit")
        elif "ray" in str(e).lower():
            print("\n💡 Tip: Install distributed computing dependencies:")
            print("   pip install ray")
        elif "rdflib" in str(e).lower():
            print("\n💡 Tip: Install semantic web dependencies:")
            print("   pip install rdflib")
        elif "openai" in str(e).lower() or "api" in str(e).lower():
            print("\n💡 Tip: Set up foundation model API keys:")
            print("   export OPENAI_API_KEY='your-key-here'")
    
    finally:
        # Cleanup
        print("\n🧹 Shutting down framework...")
        await framework.shutdown()
        print("✅ Shutdown complete!")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    dependencies = [
        ("numpy", "Scientific computing"),
        ("torch", "PyTorch deep learning"),
        ("tensorflow", "TensorFlow machine learning"),
        ("qiskit", "Quantum computing"),
        ("ray", "Distributed computing"),
        ("rdflib", "Semantic web/RDF"),
        ("fastapi", "Web API framework"),
        ("yaml", "Configuration files")
    ]
    
    missing = []
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {module} - {description}")
        except ImportError:
            print(f"  ❌ {module} - {description} (MISSING)")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print(f"💡 Install with: pip install {' '.join(missing)}")
        print("   Or: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available!")
        return True


def show_api_setup():
    """Show API key setup instructions."""
    print("\n🔑 API Key Setup (Optional):")
    
    api_keys = [
        ("OPENAI_API_KEY", "OpenAI GPT models"),
        ("ANTHROPIC_API_KEY", "Anthropic Claude models"),
        ("GOOGLE_API_KEY", "Google Gemini models"),
        ("WANDB_API_KEY", "Weights & Biases monitoring")
    ]
    
    for env_var, description in api_keys:
        value = os.getenv(env_var)
        if value:
            print(f"  ✅ {env_var} - {description} (SET)")
        else:
            print(f"  ⚪ {env_var} - {description} (optional)")
    
    print("\nNote: NSAF can run without API keys using fallback implementations.")


if __name__ == "__main__":
    print("🧪 NSAF Unified Framework Demo")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Show API setup
    show_api_setup()
    
    if deps_ok:
        print(f"\n🚀 Starting NSAF demo...")
        asyncio.run(main())
    else:
        print(f"\n❌ Please install missing dependencies first.")
        print(f"📦 Run: pip install -r requirements.txt")