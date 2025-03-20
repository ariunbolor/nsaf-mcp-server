# Installation Guide for NSAF Prototype

This guide provides instructions for installing and setting up the Neuro-Symbolic Autonomy Framework (NSAF) prototype.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nsaf_prototype.git
cd nsaf_prototype
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install the Package

#### Option 1: Install in Development Mode

This option is recommended for development as it allows you to modify the code without reinstalling.

```bash
pip install -e .
```

#### Option 2: Install from Requirements File

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

Run the tests to verify that the installation was successful:

```bash
pytest tests/
```

## Running the Examples

To run the example script:

```bash
python main.py
```

This will demonstrate the Self-Constructing Meta-Agents (SCMA) component of the NSAF framework.

## Troubleshooting

### TensorFlow Installation Issues

If you encounter issues with TensorFlow installation:

1. Make sure you have the latest pip version:
   ```bash
   pip install --upgrade pip
   ```

2. For GPU support, ensure you have the appropriate CUDA and cuDNN versions installed.

3. Consider installing TensorFlow separately:
   ```bash
   pip install tensorflow
   ```

### Other Issues

If you encounter any other issues, please check the following:

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your Python version is compatible (Python 3.8 or higher).

3. Check for any error messages and search for solutions in the TensorFlow or Python documentation.

## Next Steps

After installation, you can:

1. Explore the example code in `main.py`
2. Read the documentation in the code comments
3. Modify the configuration parameters to experiment with different settings
4. Implement your own fitness functions and datasets
