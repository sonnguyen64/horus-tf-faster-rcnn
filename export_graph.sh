#!/bin/bash

rm -r frozen_model_vgg16_horus

python tools/export_graph.py --dataset horus --net vgg16

cd ..

rm -r models/horus/1/*

cp -r faster-rcnn/frozen_model_vgg16_horus/* models/horus/1/
