#!/bin/bash
set -x  # Enable debugging output

# Clean up any existing git process locks
rm -f .git/index.lock

# Remove existing Git repository and initialize a fresh one
rm -rf .git
git init

# Setup Git LFS
git lfs install

# Create .gitattributes
echo "*.pth filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text" > .gitattributes

# Create .gitignore for large files
cat > .gitignore << EOF
# Large files over 100MB
MOSI/aligned_50.pkl
MOSI/unaligned_50.pkl
pretrained_models/self_mm-mosi_clean.pth
pretrained_models/self_mm-mosi_converted.pth
pretrained_models/self_mm-mosi_fixed.pth
pretrained_models/self_mm-mosi_matched.pth
pretrained_models/self_mm-mosi_direct_fixed.pth

# Large MP4 file and output file
Job*Interview*.mp4
output.txt

# Common ignored files
.DS_Store
__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv
env/
venv/
ENV/
.coverage
.coverage.*
.cache
EOF

# Add the remote repository
git remote add origin https://github.com/AbdullaE100/MMSA.Model.git

# List all files that would be committed
echo "Listing files to be added (excluding those in .gitignore)..."
git add -n .

# Add files excluding those in .gitignore
git add .

# Commit 
git commit -m "Add project files, excluding large files over 100MB"

# Force push to remote repository
echo "Pushing to GitHub repository..."
git push -f origin main

echo "Finished pushing files to GitHub. Large files were excluded." 