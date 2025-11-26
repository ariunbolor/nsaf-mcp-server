"""
Quantum Symbolic Library Example
-----------------------------

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Demonstrates the usage of the quantum symbolic library for various computations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.quantum_symbolic import (
    QuantumSymbolicInterface,
    SymbolicType,
    LogicalOperator,
    ComputationType
)

def demonstrate_logical_inference():
    """Demonstrate logical inference using quantum symbolic computation"""
    print("\n=== Logical Inference Example ===")
    
    # Initialize interface
    interface = QuantumSymbolicInterface()
    
    # Create variables
    a = interface.create_expression(SymbolicType.VARIABLE, "a")
    b = interface.create_expression(SymbolicType.VARIABLE, "b")
    
    # Create logical expression: a AND b
    expr = interface.create_logical_expression(
        LogicalOperator.AND,
        [a, b]
    )
    
    # Evaluate expression
    result = interface.evaluate_expression(
        expr,
        ComputationType.LOGICAL_INFERENCE,
        parameters={'shots': 1000}
    )
    
    # Print results
    print(f"Expression: {expr}")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Probability: {result.probability}")
    if result.metadata:
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

def demonstrate_constraint_solving():
    """Demonstrate constraint solving using quantum symbolic computation"""
    print("\n=== Constraint Solving Example ===")
    
    # Initialize interface
    interface = QuantumSymbolicInterface()
    
    # Create variables
    x = interface.create_expression(SymbolicType.VARIABLE, "x")
    y = interface.create_expression(SymbolicType.VARIABLE, "y")
    z = interface.create_expression(SymbolicType.VARIABLE, "z")
    
    # Create constraints: (x OR y) AND (NOT z)
    expr1 = interface.create_logical_expression(
        LogicalOperator.OR,
        [x, y]
    )
    expr2 = interface.create_logical_expression(
        LogicalOperator.NOT,
        [z]
    )
    final_expr = interface.create_logical_expression(
        LogicalOperator.AND,
        [expr1, expr2]
    )
    
    # Evaluate expression
    result = interface.evaluate_expression(
        final_expr,
        ComputationType.CONSTRAINT_SOLVING,
        parameters={'max_iterations': 100}
    )
    
    # Print results
    print(f"Expression: {final_expr}")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Probability: {result.probability}")
    if result.metadata:
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

def demonstrate_optimization():
    """Demonstrate optimization using quantum symbolic computation"""
    print("\n=== Optimization Example ===")
    
    # Initialize interface
    interface = QuantumSymbolicInterface()
    
    # Create variables for optimization problem
    q1 = interface.create_expression(SymbolicType.VARIABLE, "q1")
    q2 = interface.create_expression(SymbolicType.VARIABLE, "q2")
    q3 = interface.create_expression(SymbolicType.VARIABLE, "q3")
    
    # Create optimization expression
    expr = interface.create_logical_expression(
        LogicalOperator.AND,
        [
            interface.create_logical_expression(
                LogicalOperator.OR,
                [q1, q2]
            ),
            interface.create_logical_expression(
                LogicalOperator.XOR,
                [q2, q3]
            )
        ]
    )
    
    # Evaluate expression
    result = interface.evaluate_expression(
        expr,
        ComputationType.OPTIMIZATION,
        parameters={
            'objective': 'minimize',
            'max_iterations': 1000
        }
    )
    
    # Print results
    print(f"Expression: {expr}")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Probability: {result.probability}")
    if result.metadata:
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

def demonstrate_circuit_statistics():
    """Demonstrate circuit statistics functionality"""
    print("\n=== Circuit Statistics Example ===")
    
    # Initialize interface
    interface = QuantumSymbolicInterface()
    
    # Create a simple circuit through symbolic expression
    a = interface.create_expression(SymbolicType.VARIABLE, "a")
    b = interface.create_expression(SymbolicType.VARIABLE, "b")
    expr = interface.create_logical_expression(
        LogicalOperator.AND,
        [a, b]
    )
    
    # Convert to circuit
    circuit = interface.symbolic_processor.to_quantum_circuit(expr)
    
    # Get statistics
    stats = interface.get_circuit_statistics(circuit)
    
    # Print statistics
    print("\nCircuit Statistics:")
    print(f"Number of qubits: {stats['num_qubits']}")
    print(f"Circuit depth: {stats['circuit_depth']}")
    print("\nGate counts:")
    for gate_type, count in stats['gate_counts'].items():
        print(f"  {gate_type}: {count}")
    print(f"\nEstimated runtime: {stats['estimated_runtime']} microseconds")
    
    # Draw circuit
    print("\nCircuit visualization:")
    print(circuit.draw())

def main():
    """Main function demonstrating various quantum symbolic computations"""
    print("Quantum Symbolic Library Examples")
    print("================================")
    
    try:
        demonstrate_logical_inference()
        demonstrate_constraint_solving()
        demonstrate_optimization()
        demonstrate_circuit_statistics()
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return 1
        
    print("\nAll examples completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
