#!/bin/bash
mkdir model
wget https://storage.googleapis.com/vqamodel-mathaix/checkpoint2.pt -o model/checkpoint2.pt
wget https://storage.googleapis.com/vqamodel-mathaix/vqa.tar.gz
tar -xvzf vqa.tar.gz