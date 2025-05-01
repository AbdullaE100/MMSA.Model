#!/bin/bash

# Clean up any existing git process locks
rm -f .git/index.lock

# Setup Git LFS
git lfs install

# Create .gitattributes if it doesn't exist
echo "*.pth filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text" > .gitattributes

# Create .gitignore for large files
cat > .gitignore << EOF
# Large files over 100MB
/MOSI/aligned_50.pkl
/MOSI/unaligned_50.pkl
/pretrained_models/self_mm-mosi_clean.pth
/pretrained_models/self_mm-mosi_converted.pth
/pretrained_models/self_mm-mosi_fixed.pth
/pretrained_models/self_mm-mosi_matched.pth
/pretrained_models/self_mm-mosi_direct_fixed.pth

# Large MP4 file and output file
/Job\ Interview\ \ \ Good\ Example\ copy-2.mp4
/output.txt

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

# Initialize git if needed
if [ ! -d .git ]; then
  git init
fi

# Add the remote repository if it doesn't exist
if ! git remote | grep -q "origin"; then
  git remote add origin https://github.com/AbdullaE100/MMSA.Model.git
fi

# Set git pull to merge by default
git config pull.rebase false

# Add and commit local changes
git add .
git commit -m "Add project files, excluding large files over 100MB"

# Try to pull remote changes (with allow-unrelated-histories in case of different history)
git pull origin main --allow-unrelated-histories || echo "No remote commits to pull"

# Force push to overwrite remote repository
echo "Pushing to GitHub repository..."
git push -f origin main

echo "Finished pushing files to GitHub. Large files were excluded." 