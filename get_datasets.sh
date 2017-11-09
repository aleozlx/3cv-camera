#!/bin/bash

DATASET="02-camera.tgz"
DATASET_URL="https://storage.googleapis.com/three-cv-research-datasets/cs8680-3dcv/$DATASET"

[ -f "$DATASET" ] || wget "$DATASET_URL" && \
[ -d checkerboard-opencv ] || tar xvf "$DATASET" && \
echo "Done."
