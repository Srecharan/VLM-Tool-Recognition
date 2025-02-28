# Dataset Class Distribution Analysis
=====================================

This document provides a detailed analysis of class distributions across the three datasets used in our Vision Language Model (VLM) project.

## Raw Dataset Distributions
---------------------------

The following tables show the original class distributions in each dataset:

### Dataset 1

| Class | Count |
|-------|-------|
| Combination wrench | 557 |
| Screwdriver | 446 |
| Phillips screwdriver | 393 |
| Claw hammer | 337 |
| Open-End wrench | 181 |
| Allen wrench | 154 |
| Precision screwdriver | 149 |
| Nose pliers | 129 |
| Interchangeable screwdriver | 113 |
| Box-End wrench | 103 |
| Adjustable wrench | 102 |
| Diagonal pliers | 101 |
| Ball-peen hammer | 97 |
| Socket wrench | 79 |
| Ratchering wrench | 76 |
| Torx screwdriver | 64 |
| Tongue and groove pliers | 54 |
| Linesman pliers | 52 |
| Locking pliers | 45 |
| Combination pliers | 25 |
| Allen | 24 |
| Pipe wrench | 19 |
| Square screwdriver | 13 |
| Torx wrench | 12 |
| Pozidriv screwdriver | 9 |
| Slip joint pliers | 9 |
| Star screwdriver | 3 |
| Interchangeable | 1 |
| Needle nose pliers | 1 |
| Open-End | 1 |
| Spanner screwdriver | 1 |

### Dataset 2

| Class | Count |
|-------|-------|
| Screwdriver | 3,556 |
| Wrench | 3,462 |
| Pliers | 3,338 |
| Screw | 3,308 |
| Bolt | 3,303 |
| Pebble | 3,290 |
| Hammer | 193 |
| Adjustable Wrench | 142 |
| Needle Nose Pliers | 126 |
| Diagonal Pliers | 56 |
| Tool Box | 48 |
| Line Mans Pliers | 28 |
| Dynanometer | 24 |
| Slip Joint Pliers | 21 |
| Tester | 17 |
| Nails | 9 |
| Mallet | 7 |
| Hand Saw | 5 |
| Knife | 5 |
| Back Saw | 4 |
| Brace | 3 |
| Axe | 1 |
| Bradawl | 1 |
| Clip | 1 |

### Dataset 3

| Class | Count |
|-------|-------|
| Screwdriver | 3,112 |
| Wrench | 1,710 |
| Adjustable spanner | 1,302 |
| Hammer | 1,284 |
| Needle-nose pliers | 566 |
| Ratchet | 515 |
| Pliers | 492 |
| Gas wrench | 440 |
| Cutting pliers | 316 |
| Hand | 209 |
| Tape measure | 199 |
| Utility knife | 178 |
| Calipers | 170 |
| Backsaw | 96 |
| Drill | 75 |
| Handsaw | 55 |
| Gun | 46 |

## Aggregation Methodology
-------------------------

To standardize class naming across datasets, we aggregated similar tool types:

| Object Category | Dataset 1 Calculation | Dataset 2 Calculation | Dataset 3 Calculation |
|-----------------|------------------------|------------------------|------------------------|
| Screwdriver | 446 (Screwdriver) + 393 (Phillips screwdriver) + 149 (Precision screwdriver) + 113 (Interchangeable screwdriver) + 64 (Torx screwdriver) + 13 (Square screwdriver) + 9 (Pozidriv screwdriver) + 3 (Star screwdriver) + 1 (Spanner screwdriver) | 3556 (Screwdriver) | 3112 (Screwdriver) |
| Wrench | 557 (Combination wrench) + 181 (Open-End wrench) + 154 (Allen wrench) + 103 (Box-End wrench) + 102 (Adjustable wrench) + 79 (Socket wrench) + 76 (Ratchering wrench) + 24 (Allen) + 19 (Pipe wrench) + 12 (Torx wrench) + 1 (Open-End) | 3462 (Wrench) + 142 (Adjustable Wrench) | 1710 (Wrench) + 1302 (Adjustable spanner) |
| Pliers | 129 (Nose pliers) + 101 (Diagonal pliers) + 54 (Tongue and groove pliers) + 52 (Linesman pliers) + 45 (Locking pliers) + 25 (Combination pliers) + 9 (Slip joint pliers) + 1 (Needle nose pliers) | 3338 (Pliers) + 126 (Needle Nose Pliers) + 56 (Diagonal Pliers) + 28 (Line Mans Pliers) + 21 (Slip Joint Pliers) | 566 (Needle-nose pliers) + 492 (Pliers) + 316 (Cutting pliers) |
| Hammer | 337 (Claw hammer) + 97 (Ball-peen hammer) | 193 (Hammer) | 1284 (Hammer) |

## Final Aggregated Category Counts
---------------------------------

After aggregation, the distribution of tool categories across datasets is as follows:

| Object Category | Dataset 1 | Dataset 2 | Dataset 3 |
|-----------------|-----------|-----------|-----------|
| Wrench | 1036 | 3604 | 3012 |
| Hammer | 434 | 193 | 1284 |
| Pliers | 359 | 3541 | 808 |
| Screwdriver | 1170 | 3556 | 3112 |
| Bolt | 0 | 3303 | 0 |
| Dynanometer | 0 | 24 | 0 |
| Tester | 0 | 17 | 0 |
| Tool Box | 0 | 48 | 0 |
| Tape measure | 0 | 0 | 199 |
| Ratchet | 0 | 0 | 515 |
| Drill | 0 | 0 | 75 |
| Calipers | 0 | 0 | 170 |
| Saw | 0 | 4 | 96 |

## Filtered Object Category Counts
---------------------------------

| Object Category | Dataset 1 | Dataset 2 | Dataset 3 |
|-----------------|-----------|-----------|-----------|
| Wrench | 1036 | 3604 | 3012 |
| Hammer | 434 | 193 | 1284 |
| Pliers | 359 | 3541 | 808 |
| Screwdriver | 1170 | 3556 | 3112 |
| Bolt | 0 | 3303 | 0 |
| Dynanometer | 0 | 24 | 0 |
| Tester | 0 | 17 | 0 |
| Tool Box | 0 | 48 | 0 |
| Tape measure | 0 | 0 | 199 |
| Ratchet | 0 | 0 | 515 |
| Drill | 0 | 0 | 75 |
| Calipers | 0 | 0 | 170 |
| Saw | 0 | 4 | 96 |

## References
------------

*   Dataset 1: <https://universe.roboflow.com/centro-de-enseanza-tcnica-industrial/tools-v1/health>
*   Dataset 2: <https://universe.roboflow.com/gopi-cheetah-p7w4o/tools-tx9kq/health>
*   Dataset 3: <https://universe.roboflow.com/objectdetection-w4liy/object_detection_2-j9tgq/health>
