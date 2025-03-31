#!/bin/bash

# Update the package list
apt-get update 

# Install Tesseract-OCR
apt-get install -y tesseract-ocr 

# Start the application
python check.py
