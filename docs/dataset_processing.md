# Dataset Processing Guide

This guide explains how to process and upload the tool detection dataset from Roboflow to HuggingFace, including data organization and processing steps.

## Installation

Install the required Python packages:
```bash
pip install -r req.txt
```

Required packages:
- pandas: Data manipulation and analysis
- pillow: Image processing
- datasets: HuggingFace datasets library
- huggingface-hub: HuggingFace hub interface

## Folder Structure

After processing, your dataset directory should look like this:

```
dataset/
├── images/ # Contains all images
│ ├── train/ # Training set images
│ ├── valid/ # Validation set images
│ └── test/ # Test set images
├── xml/ # Contains processed XML annotations
│ ├── train/ # Training set annotations
│ ├── valid/ # Validation set annotations
│ └── test/ # Test set annotations
├── tool_use.xml # Tool usage and safety information
├── download_from_roboflow.sh
├── consolidate_xml.py
├── add_tool_use.py
└── process.py
```

## Processing Steps

### 1. Download Dataset from Roboflow

First, download and extract the dataset from Roboflow:

```bash
# Execute the download script
chmod +x download_from_roboflow.sh
./download_from_roboflow.sh
```

This will:
- Download the dataset as `roboflow.zip`
- Extract it to a `roboflow` directory

### 2. Process XML Annotations

Run the XML consolidation script to clean and organize annotations:
```bash
python consolidate_xml.py
```

This script:
- Reads XML files from the roboflow directory
- Creates cleaned versions in the `xml` directory
- Organizes them into train/valid/test splits

### 3. Organize Files

Remove XML files from image directories to maintain clean separation:
```bash
find /path/to/dataset/roboflow -type f -name "*.xml" -exec rm {} +
mv /path/to/dataset/roboflow /path/to/dataset/images
```

This ensures:
- Clean separation between images and annotations
- Proper directory structure for subsequent processing

### 4. Add Tool Usage Information

Add safety and usage information to the annotations:
```bash
python add_tool_use.py
```

This step:
- Reads the `tool_use.xml` file
- Adds detailed tool information to each annotation
- Maintains the split organization

### 5. Process and Upload to HuggingFace

Finally, process the dataset and upload to HuggingFace:
```bash
python process.py
```

This script:
- Combines images with their enhanced annotations
- Creates preview files for each split
- Uploads the processed dataset to HuggingFace (optional)

## Notes

- Ensure you have all required Python packages installed
- Make sure `tool_use.xml` is present in the dataset directory
- The HuggingFace token should be configured in `process.py`
- Preview files will be generated for each split to verify the data structure

