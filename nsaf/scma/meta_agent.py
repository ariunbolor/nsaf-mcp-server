"""
Meta-Agent implementation for the Self-Constructing Meta-Agents (SCMA) module.
"""

import uuid
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple, Union

from ..utils.config import Config


class MetaAgent:
    """
    Base class for a self-constructing meta-agent.
    
    A meta-agent is an AI agent that can self-design and evolve,
    creating optimized versions of itself dynamically.
    """
    
    def __init__(self, config: Optional[Config] = None, architecture: Optional[Dict[str, Any]] = None):
        """
        Initialize a new MetaAgent instance.
        
        Args:
            config: Configuration object containing parameters for the agent.
                   If None, default configuration will be used.
            architecture: Optional dictionary describing the neural architecture of the agent.
                         If None, a random architecture will be generated.
        """
        self.config = config or Config()
        self.id = str(uuid.uuid4())
        self.architecture = architecture or self._generate_random_architecture()
        self.model = self._build_model()
        self.fitness = 0.0
        self.generation = 0
        self.parent_ids = []
        self.creation_timestamp = tf.timestamp().numpy()
        self.metadata = {}
        
    def _generate_random_architecture(self) -> Dict[str, Any]:
        """
        Generate a random neural network architecture for the agent.
        
        Returns:
            A dictionary describing the neural architecture.
        """
        complexity = self.config.get('scma.architecture_complexity', 'medium')
        
        # Define complexity levels
        if complexity == 'simple':
            max_layers = 2
            max_neurons = 64
            max_connections = 128
        elif complexity == 'medium':
            max_layers = 4
            max_neurons = 128
            max_connections = 512
        else:  # complex
            max_layers = 8
            max_neurons = 256
            max_connections = 1024
        
        # Generate random architecture
        num_layers = np.random.randint(1, max_layers + 1)
        hidden_layers = []
        
        for _ in range(num_layers):
            layer_size = np.random.randint(16, max_neurons + 1)
            hidden_layers.append(layer_size)
        
        # Select random activation functions for each layer
        activation_options = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']
        activations = np.random.choice(activation_options, size=num_layers).tolist()
        
        # Determine if to use batch normalization and dropout
        use_batch_norm = np.random.choice([True, False])
        dropout_rate = np.random.uniform(0, 0.5) if np.random.choice([True, False]) else 0
        
        # Select optimizer and learning rate
        optimizer_options = ['adam', 'sgd', 'rmsprop', 'adagrad']
        optimizer = np.random.choice(optimizer_options)
        learning_rate = np.random.uniform(0.0001, 0.01)
        
        # Generate connection patterns (for more complex architectures)
        connections = {}
        if complexity != 'simple':
            num_connections = np.random.randint(num_layers, min(max_connections, num_layers * (num_layers - 1) // 2 + num_layers))
            for _ in range(num_connections):
                from_layer = np.random.randint(0, num_layers)
                to_layer = np.random.randint(from_layer + 1, num_layers + 1)
                if to_layer < num_layers:  # Skip output layer for internal connections
                    connection_key = f"{from_layer}->{to_layer}"
                    connections[connection_key] = np.random.uniform(0.1, 1.0)  # Connection strength
        
        return {
            'hidden_layers': hidden_layers,
            'activations': activations,
            'use_batch_normalization': use_batch_norm,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'connections': connections,
            'input_shape': (None, self.config.get('scma.agent_memory_size', 1024)),
            'output_shape': (None, self.config.get('scma.agent_memory_size', 1024)),
        }
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build a TensorFlow model based on the agent's architecture.
        
        Returns:
            A compiled TensorFlow model.
        """
        # Extract architecture parameters
        hidden_layers = self.architecture['hidden_layers']
        activations = self.architecture['activations']
        use_batch_norm = self.architecture['use_batch_normalization']
        dropout_rate = self.architecture['dropout_rate']
        input_shape = self.architecture['input_shape'][1]  # Get the second dimension (memory size)
        output_shape = self.architecture['output_shape'][1]
        
        # Create model
        inputs = tf.keras.Input(shape=(input_shape,))
        x = inputs
        
        # Build hidden layers
        for i, (units, activation) in enumerate(zip(hidden_layers, activations)):
            x = tf.keras.layers.Dense(units)(x)
            
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
                
            x = tf.keras.layers.Activation(activation)(x)
            
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(output_shape, activation='linear')(x)
        
        # Create and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Configure optimizer
        optimizer_name = self.architecture['optimizer']
        learning_rate = self.architecture['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:  # adagrad
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the agent's model.
        
        Args:
            inputs: Input data for the model.
            
        Returns:
            Model predictions.
        """
        return self.model.predict(inputs)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the agent's model on the provided data.
        
        Args:
            x_train: Training input data.
            y_train: Training target data.
            x_val: Optional validation input data.
            y_val: Optional validation target data.
            
        Returns:
            Dictionary containing training history.
        """
        batch_size = self.config.get('training.batch_size', 32)
        epochs = self.config.get('training.epochs', 100)
        validation_split = self.config.get('training.validation_split', 0.2)
        
        callbacks = []
        
        # Add early stopping if configured
        early_stopping_patience = self.config.get('training.early_stopping_patience')
        if early_stopping_patience:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Add model checkpoint if configured
        if self.config.get('training.model_checkpoint', False):
            checkpoint_path = self.config.get('training.checkpoint_path', './checkpoints')
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{checkpoint_path}/agent_{self.id}_{{epoch:02d}}_{{val_loss:.4f}}.h5",
                monitor='val_loss',
                save_best_only=True
            )
            callbacks.append(model_checkpoint)
        
        # Train the model
        if x_val is not None and y_val is not None:
            history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks
            )
        
        return history.history
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the agent's model on test data.
        
        Args:
            x_test: Test input data.
            y_test: Test target data.
            
        Returns:
            Tuple of (loss, metric) values.
        """
        return self.model.evaluate(x_test, y_test)
    
    def mutate(self, mutation_rate: Optional[float] = None) -> 'MetaAgent':
        """
        Create a mutated copy of this agent.
        
        Args:
            mutation_rate: Probability of mutation for each parameter.
                          If None, the value from config will be used.
                          
        Returns:
            A new MetaAgent with mutated architecture.
        """
        if mutation_rate is None:
            mutation_rate = self.config.get('scma.mutation_rate', 0.1)
        
        # Create a copy of the architecture
        new_architecture = self.architecture.copy()
        
        # Mutate hidden layers
        if np.random.random() < mutation_rate:
            hidden_layers = new_architecture['hidden_layers'].copy()
            
            # Randomly add, remove, or modify a layer
            mutation_type = np.random.choice(['add', 'remove', 'modify'])
            
            if mutation_type == 'add' and len(hidden_layers) < 8:
                # Add a new layer at a random position
                position = np.random.randint(0, len(hidden_layers) + 1)
                layer_size = np.random.randint(16, 256)
                hidden_layers.insert(position, layer_size)
                
                # Add a corresponding activation function
                activation_options = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']
                new_activation = np.random.choice(activation_options)
                new_architecture['activations'].insert(position, new_activation)
                
            elif mutation_type == 'remove' and len(hidden_layers) > 1:
                # Remove a random layer
                position = np.random.randint(0, len(hidden_layers))
                hidden_layers.pop(position)
                new_architecture['activations'].pop(position)
                
            elif mutation_type == 'modify' and len(hidden_layers) > 0:
                # Modify a random layer
                position = np.random.randint(0, len(hidden_layers))
                # Change size by up to 50% in either direction
                current_size = hidden_layers[position]
                change_factor = np.random.uniform(0.5, 1.5)
                new_size = max(16, min(256, int(current_size * change_factor)))
                hidden_layers[position] = new_size
            
            new_architecture['hidden_layers'] = hidden_layers
        
        # Mutate activation functions
        if np.random.random() < mutation_rate:
            activation_options = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']
            position = np.random.randint(0, len(new_architecture['activations']))
            new_architecture['activations'][position] = np.random.choice(activation_options)
        
        # Mutate batch normalization
        if np.random.random() < mutation_rate:
            new_architecture['use_batch_normalization'] = not new_architecture['use_batch_normalization']
        
        # Mutate dropout rate
        if np.random.random() < mutation_rate:
            if new_architecture['dropout_rate'] > 0:
                # Either modify or remove dropout
                if np.random.choice([True, False]):
                    new_architecture['dropout_rate'] = np.random.uniform(0, 0.5)
                else:
                    new_architecture['dropout_rate'] = 0
            else:
                # Add dropout
                new_architecture['dropout_rate'] = np.random.uniform(0, 0.5)
        
        # Mutate optimizer
        if np.random.random() < mutation_rate:
            optimizer_options = ['adam', 'sgd', 'rmsprop', 'adagrad']
            new_architecture['optimizer'] = np.random.choice(optimizer_options)
        
        # Mutate learning rate
        if np.random.random() < mutation_rate:
            # Change learning rate by up to one order of magnitude in either direction
            current_lr = new_architecture['learning_rate']
            change_factor = np.random.uniform(0.1, 10)
            new_lr = max(0.0001, min(0.1, current_lr * change_factor))
            new_architecture['learning_rate'] = new_lr
        
        # Create a new agent with the mutated architecture
        child = MetaAgent(config=self.config, architecture=new_architecture)
        child.parent_ids = [self.id]
        child.generation = self.generation + 1
        
        return child
    
    def crossover(self, other: 'MetaAgent') -> 'MetaAgent':
        """
        Perform crossover with another agent to create a child.
        
        Args:
            other: Another MetaAgent to crossover with.
            
        Returns:
            A new MetaAgent with combined architecture from both parents.
        """
        # Create a new architecture by combining elements from both parents
        new_architecture = {}
        
        # Crossover hidden layers and activations
        # We'll take some layers from each parent
        if len(self.architecture['hidden_layers']) > 0 and len(other.architecture['hidden_layers']) > 0:
            # Determine crossover points for each parent
            self_layers = self.architecture['hidden_layers']
            other_layers = other.architecture['hidden_layers']
            
            self_activations = self.architecture['activations']
            other_activations = other.architecture['activations']
            
            # Randomly choose which parent to start with
            if np.random.choice([True, False]):
                # Start with self, end with other
                crossover_point_self = np.random.randint(1, len(self_layers))
                crossover_point_other = np.random.randint(0, len(other_layers))
                
                new_layers = self_layers[:crossover_point_self] + other_layers[crossover_point_other:]
                new_activations = self_activations[:crossover_point_self] + other_activations[crossover_point_other:]
            else:
                # Start with other, end with self
                crossover_point_other = np.random.randint(1, len(other_layers))
                crossover_point_self = np.random.randint(0, len(self_layers))
                
                new_layers = other_layers[:crossover_point_other] + self_layers[crossover_point_self:]
                new_activations = other_activations[:crossover_point_other] + self_activations[crossover_point_self:]
            
            new_architecture['hidden_layers'] = new_layers
            new_architecture['activations'] = new_activations
        else:
            # If one parent has no layers, use the other parent's layers
            parent_with_layers = self if len(self.architecture['hidden_layers']) > 0 else other
            new_architecture['hidden_layers'] = parent_with_layers.architecture['hidden_layers'].copy()
            new_architecture['activations'] = parent_with_layers.architecture['activations'].copy()
        
        # For other parameters, randomly choose from either parent
        for key in ['use_batch_normalization', 'dropout_rate', 'optimizer', 'learning_rate']:
            new_architecture[key] = self.architecture[key] if np.random.choice([True, False]) else other.architecture[key]
        
        # For connections, combine from both parents
        new_connections = {}
        for key in set(list(self.architecture.get('connections', {}).keys()) + 
                      list(other.architecture.get('connections', {}).keys())):
            if key in self.architecture.get('connections', {}) and key in other.architecture.get('connections', {}):
                # If both parents have this connection, take average or choose one
                if np.random.choice([True, False]):
                    new_connections[key] = (self.architecture['connections'][key] + 
                                           other.architecture['connections'][key]) / 2
                else:
                    new_connections[key] = (self.architecture['connections'][key] if 
                                           np.random.choice([True, False]) else 
                                           other.architecture['connections'][key])
            elif key in self.architecture.get('connections', {}):
                # Only self has this connection
                if np.random.random() < 0.7:  # 70% chance to inherit
                    new_connections[key] = self.architecture['connections'][key]
            else:
                # Only other has this connection
                if np.random.random() < 0.7:  # 70% chance to inherit
                    new_connections[key] = other.architecture['connections'][key]
        
        new_architecture['connections'] = new_connections
        
        # Keep the same input and output shapes
        new_architecture['input_shape'] = self.architecture['input_shape']
        new_architecture['output_shape'] = self.architecture['output_shape']
        
        # Create a new agent with the combined architecture
        child = MetaAgent(config=self.config, architecture=new_architecture)
        child.parent_ids = [self.id, other.id]
        child.generation = max(self.generation, other.generation) + 1
        
        return child
    
    def save(self, filepath: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            filepath: Path to save the agent.
        """
        # Save the model
        self.model.save(f"{filepath}_model.h5")
        
        # Save agent metadata
        agent_data = {
            'id': self.id,
            'architecture': self.architecture,
            'fitness': self.fitness,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'creation_timestamp': self.creation_timestamp,
            'metadata': self.metadata
        }
        
        np.save(f"{filepath}_data.npy", agent_data)
    
    @classmethod
    def load(cls, filepath: str, config: Optional[Config] = None) -> 'MetaAgent':
        """
        Load an agent from a file.
        
        Args:
            filepath: Path to load the agent from.
            config: Optional configuration object.
            
        Returns:
            The loaded MetaAgent.
        """
        # Load agent metadata
        agent_data = np.load(f"{filepath}_data.npy", allow_pickle=True).item()
        
        # Create a new agent with the loaded architecture
        agent = cls(config=config, architecture=agent_data['architecture'])
        
        # Load the model
        agent.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        
        # Set agent properties
        agent.id = agent_data['id']
        agent.fitness = agent_data['fitness']
        agent.generation = agent_data['generation']
        agent.parent_ids = agent_data['parent_ids']
        agent.creation_timestamp = agent_data['creation_timestamp']
        agent.metadata = agent_data['metadata']
        
        return agent
    
    def __str__(self) -> str:
        """
        Get a string representation of the agent.
        
        Returns:
            A string describing the agent.
        """
        return (f"MetaAgent(id={self.id}, "
                f"fitness={self.fitness:.4f}, "
                f"generation={self.generation}, "
                f"layers={self.architecture['hidden_layers']})")
