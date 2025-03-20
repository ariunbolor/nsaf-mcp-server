#!/bin/bash

# Setup script for deploying NSAF MCP Server to GitHub

# Check if GitHub username is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <github-username>"
  echo "Example: $0 yourusername"
  exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME="nsaf-mcp-server"
GITHUB_REPO="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo "Setting up NSAF MCP Server for GitHub deployment..."
echo "GitHub Repository: $GITHUB_REPO"

# Initialize git repository
git init
git add .
git commit -m "Initial commit"
git branch -M main

# Add GitHub remote
echo "Adding GitHub remote..."
git remote add origin $GITHUB_REPO

echo ""
echo "Setup complete! To push to GitHub, run:"
echo "git push -u origin main"
echo ""
echo "After pushing to GitHub, you can install the MCP server globally with:"
echo "npm install -g $GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "Then add it to your MCP settings configuration as shown in README.md"
