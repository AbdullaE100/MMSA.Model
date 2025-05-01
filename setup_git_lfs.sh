#!/bin/zsh

# Navigate to the project directory
cd /Users/abdullaehsan/Desktop/FINALproject/MMSA.Model

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is not installed. Installing with Homebrew..."
    brew install git-lfs
fi

# Initialize Git LFS
echo "Setting up Git LFS for your user account..."
git lfs install

# Track large files (PSD files as an example, but you can modify this for your specific file types)
echo "Setting up tracking for large files..."
git lfs track "*.psd"
git lfs track "*.mp4"
git lfs track "*.h5"
git lfs track "*.model"

# Make sure .gitattributes is tracked
echo "Making sure .gitattributes is tracked..."
git add .gitattributes

# Check remote repository configuration
echo "Checking remote repository configuration..."
if ! git remote -v | grep -q "AbdullaE100/MMSA.Model"; then
    echo "Setting up remote repository..."
    git remote add origin https://github.com/AbdullaE100/MMSA.Model.git
fi

echo "Git LFS setup complete!"
echo "Now you can add, commit, and push your files using:"
echo "git add file.psd"
echo "git commit -m \"Add design file\""
echo "git push origin main" 