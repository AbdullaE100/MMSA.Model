#!/bin/zsh

# Navigate to the repository
cd /Users/abdullaehsan/Desktop/FINALproject/MMSA.Model

# Reset any pending changes
git reset

# Add only the necessary files
git add README.md .gitattributes

# Commit the changes
git commit -m "Configure Git LFS and add README"

# Push to GitHub
git push origin main

echo "Completed pushing Git LFS configuration to GitHub!"
echo "Now you can selectively add your large files using:"
echo "git add your_large_file.mp4"
echo "git commit -m \"Add large file\""
echo "git push origin main" 