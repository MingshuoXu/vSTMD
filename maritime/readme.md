
# Lightweight Motion-Based Tiny Object Detection for Maritime Search & Rescue (SeaDronesSee)

This project proposes a lightweight video motion detection model tailored for extremely tiny human targets in maritime environments.
It is designed for real-time search and rescue (SAR) under the see-via Visual Aerial Unit (VAU) scenario, where speed, robustness, and low computational cost are crucial.

Unlike appearance-based detectors that struggle with tiny objects and low contrast, the proposed method leverages motion cues extracted from video sequences, enabling robust detection of people in the water even under challenging UAV flight conditions.

### Dataset

SeaDronesSee Detection & Tracking Dataset
- Homepage: https://seadronessee.cs.uni-tuebingen.de/home

- SeaDronesSee provides UAV maritime surveillance footage with annotated humans in the water, including many scenarios with tiny, low-contrast, or partially occluded targets.

### Demo Videos

We provide several demonstration clips from the SeaDronesSee dataset to showcase the performance and robustness of the proposed motion detection model.

1. demo1-SeaDronesSee-696-1410.mp4

- Twelve people in the water

- Demonstrates multi-target detection on tiny-scale targets

2. demo2-SeaDronesSee-1697-2697.mp4

- Single person in the water

- Clear tiny-target detection scenario

3. demo3-SeaDronesSee-3666-4100.mp4

- Three people in the water

- Challenges:

    - UAV hovers for the first half, reducing relative motion

    - Demonstrates robustness under low-motion conditions

4. demo4-SeaDronesSee-22931-23931.mp4

- Three people in the water

- Includes hovering segments

5. demo5-SeaDronesSee-29713-30312.mp4

- Seven people in the water

- Normal small size targets

- Contains long-term UAV hovering, making detection extremely challenging







