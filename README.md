# Performance Benchmarking framework

## Overview

This framework is used to run a different workloads (originally focused on model inference), measure time of each run, and provide results in machine and human readable formats.

## Installation

### Requirements
 - Python
 - [Optional] OpenPyXL - for generating reports in Excel format
 - [Optional] Excel - for running aggregation tool

## Running

### Arguments
- --batch-size - list of batch sizes has to be verified

### Examples

Simple run of YOLO11 Large model benchmarking using ONNXRuntime with default settings, batch size is 1.

```bash
python test_perf.py models.yolo11l.ort
```

Simple run of YOLO11 Large model benchmarking using ONNXRuntime with default settings, custom set of batch size: 1, 2, 4, 8, 16.

```bash
python test_perf.py models.yolo11l.ort --batch-size 1,2,4,8,16
```

