#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install pipwin
pipwin refresh
pipwin install pyaudio
pip install -r requirements.txt
