#!/bin/bash

GITHUB_USERNAME="dolwinf@gmail.com"

GITHUB_TOKEN="$GITHUB_ACCESS_TOKEN"

cd /workspace

git clone https://${GITHUB_TOKEN}@github.com/dolwinf/stable-diffusion.git

cd stable-diff

pip install -r requirements.txt

git config --global user.email "dolwinf@gmail.com"