#!/usr/bin/env python3
"""
NSAF Framework Example
=====================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Original NSAF framework example demonstrating core functionality.
"""

import asyncio
import json
from pathlib import Path
from core.framework import NeuroSymbolicAutonomyFramework

async def main():
    # Initialize framework
    framework = NeuroSymbolicAutonomyFramework()
    
    # Example task definition
    task = {
        'description': 'Develop a machine learning model for predictive maintenance',
        'goals': [
            {
                'type': 'accuracy',
                'target': 0.95,
                'priority': 0.8
            },
            {
                'type': 'latency',
                'target': 100,  # milliseconds
                'priority': 0.7
            }
        ],
        'constraints': [
            {
                'type': 'memory',
                'limit': '4GB',
                'importance': 0.9
            },
            {
                'type': 'runtime',
                'limit': '24h',
                'importance': 0.6
            }
        ],
        'requirements': {
            'frameworks': ['pytorch', 'scikit-learn'],
            'data_sources': ['sensor_data', 'maintenance_logs'],
            'deployment': 'edge_device'
        },
        'tasks': [
            {
                'name': 'data_preprocessing',
                'type': 'pipeline',
                'priority': 0.9,
                'dependencies': []
            },
            {
                'name': 'feature_engineering',
                'type': 'analysis',
                'priority': 0.8,
                'dependencies': ['data_preprocessing']
            },
            {
                'name': 'model_training',
                'type': 'ml',
                'priority': 1.0,
                'dependencies': ['feature_engineering']
            },
            {
                'name': 'model_optimization',
                'type': 'optimization',
                'priority': 0.7,
                'dependencies': ['model_training']
            },
            {
                'name': 'deployment_preparation',
                'type': 'deployment',
                'priority': 0.6,
                'dependencies': ['model_optimization']
            }
        ],
        'complexity': 0.8
    }
    
    try:
        # Process task
        print("\nProcessing task...")
        result = await framework.process_task(task)
        
        # Print results
        print("\nTask Processing Results:")
        print(json.dumps(result, indent=2))
        
        # Example task state update
        print("\nUpdating task state...")
        state_update = {
            'progress': 0.5,
            'metrics': {
                'accuracy': 0.92,
                'latency': 95
            },
            'resource_usage': {
                'memory': '2.1GB',
                'cpu': 0.75
            },
            'status': 'training',
            'understanding': 0.85,
            'confidence': 0.8
        }
        
        update_result = await framework.update_task_state(
            result['task_clusters'][0]['id'],
            state_update
        )
        
        print("\nTask Update Results:")
        print(json.dumps(update_result, indent=2))
        
        # Get system status
        print("\nSystem Status:")
        status = framework.get_system_status()
        print(json.dumps(status, indent=2))
        
        # Get active agents
        print("\nActive Agents:")
        agents = framework.get_active_agents()
        print(json.dumps(agents, indent=2))
        
        # Get task clusters
        print("\nTask Clusters:")
        clusters = framework.get_task_clusters()
        print(json.dumps(clusters, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Shutdown framework
        print("\nShutting down framework...")
        await framework.shutdown()

if __name__ == "__main__":
    # Run example
    asyncio.run(main())
