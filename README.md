# Car Detection and Counting

A team project developed as part of the *Digital Image Processing and Analysis* course at FER (Faculty of Electrical Engineering and Computing, University of Zagreb).

## 📌 Project Overview

This project compares the performance of:
- Pre-trained YOLO models (e.g. YOLOv8)
- A custom-built object detection model

on the task of car detection and counting in road images.

## 🔧 Features
- Dataset preprocessing and conversion to YOLO format
- Evaluation of object detection performance (mAP, precision, recall, F1)
- YAML-based training configuration
- Graph and report generation
- Counting logic (static images, extendable to video)

## 📁 Structure
```
├── data/           # Raw and preprocessed dataset
├── models/         # YOLO and custom models
├── configs/        # YAML config files
├── training/       # Training & evaluation scripts
├── utils/          # Helpers for plotting, metrics, etc.
├── results/        # Metrics, logs, and visualizations
```

## 🚀 Quick Start (YOLOv8)
```bash
pip install -r requirements.txt
yolo task=detect mode=val model=yolov8n.pt data=configs/your_dataset.yaml
```

## 📅 Status
Work in progress
