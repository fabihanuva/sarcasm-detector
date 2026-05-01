#!/bin/bash
echo "Setting up Sarcasm Detector..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Setup complete! Now run: source venv/bin/activate && python3 run.py"
