#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import signal

# Make sure to kill any existing Python processes
os.system("pkill -f 'python.*gradio' || true")
time.sleep(2)

# Start the app and capture output
process = subprocess.Popen(
    ["python3", "video_sentiment_app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True
)

# File to save the link to
link_file = "gradio_link.txt"
link_found = False

with open(link_file, "w") as f:
    for i in range(30):  # 30 seconds timeout
        if process.poll() is not None:
            # Process ended
            break
            
        line = process.stdout.readline()
        if not line:
            time.sleep(1)
            continue
            
        f.write(line)
        f.flush()
        
        # Look for the gradio link
        if "gradio.live" in line or "Running on public URL" in line:
            link_found = True
            print("FOUND LINK: " + line.strip())
            break
    
    if not link_found:
        print("No Gradio link found after 30 seconds")
    
# Keep the app running in the background
if process.poll() is None:
    # Detach the process so it continues running
    os.system(f"disown {process.pid} 2>/dev/null || true")
    print(f"App is running as PID {process.pid}")
else:
    print("App process ended unexpectedly") 