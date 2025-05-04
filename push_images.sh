#!/bin/bash

# Add README and images to git
git add README.md
git add imagez/

# Commit changes
git commit -m "Add screenshots in imagez folder and update README links"

# Push to GitHub
git push origin main

# Print status
echo "Push completed!" 