#!/bin/bash

# Improved setup script for deploying NSAF MCP Server to GitHub

# Check if GitHub username is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <github-username> [--force]"
  echo "Example: $0 yourusername"
  echo "Use --force to force push (overwrites remote repository content)"
  exit 1
fi

GITHUB_USERNAME=$1
FORCE_PUSH=false

# Check for force flag
if [ "$2" == "--force" ]; then
  FORCE_PUSH=true
  echo "Force push enabled. This will overwrite any existing content in the remote repository."
fi

REPO_NAME="nsaf-mcp-server"
GITHUB_REPO="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo "Setting up NSAF MCP Server for GitHub deployment..."
echo "GitHub Repository: $GITHUB_REPO"

# Check if git is already initialized
if [ -d ".git" ]; then
  echo "Git repository already initialized."
else
  echo "Initializing git repository..."
  git init
fi

# Check if remote origin already exists
if git remote | grep -q "origin"; then
  echo "Remote 'origin' already exists."
  
  # Update the remote URL if it's different
  CURRENT_REMOTE=$(git remote get-url origin)
  if [ "$CURRENT_REMOTE" != "$GITHUB_REPO" ]; then
    echo "Updating remote URL from $CURRENT_REMOTE to $GITHUB_REPO"
    git remote set-url origin "$GITHUB_REPO"
  fi
else
  echo "Adding GitHub remote..."
  git remote add origin "$GITHUB_REPO"
fi

# Add all files
git add .

# Check if there are changes to commit
if git diff-index --quiet HEAD --; then
  echo "No changes to commit."
else
  echo "Committing changes..."
  git commit -m "Update NSAF MCP Server"
fi

# Set main branch
git branch -M main

# Handle pushing to GitHub
echo "Preparing to push to GitHub..."

if [ "$FORCE_PUSH" = true ]; then
  echo "Force pushing to GitHub (this will overwrite remote content)..."
  git push -f -u origin main
else
  # Try to pull first to integrate remote changes
  echo "Pulling from remote repository to integrate changes..."
  if git pull --rebase origin main; then
    echo "Pull successful, pushing changes..."
    git push -u origin main
  else
    echo ""
    echo "ERROR: Failed to pull from remote repository."
    echo "This could be because:"
    echo "1. The remote repository has content that conflicts with your local changes."
    echo "2. The remote repository exists but is empty (try with --force)."
    echo "3. You don't have permission to access the repository."
    echo ""
    echo "Options:"
    echo "1. Run this script with --force to overwrite remote content: ./setup-github-fixed.sh $GITHUB_USERNAME --force"
    echo "2. Manually resolve conflicts and push:"
    echo "   git pull origin main"
    echo "   # resolve conflicts"
    echo "   git add ."
    echo "   git commit -m \"Resolve conflicts\""
    echo "   git push -u origin main"
    echo ""
    exit 1
  fi
fi

echo ""
echo "Next steps:"
echo ""
echo "1. Build the MCP server:"
echo "   npm install"
echo "   npm run build"
echo ""
echo "2. Install the MCP server globally:"
echo "   npm install -g $GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "3. Add it to your MCP settings configuration as shown in README.md"
echo ""
echo "Note: If npm install -g fails, you may need to wait a few minutes for GitHub to process your push."
echo "      You can also install directly from the local directory:"
echo "      npm install -g ."
