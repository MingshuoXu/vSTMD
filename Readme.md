**vSTMD**: Visual Motion Detection for Extremely Tiny Target at Various Velocities
---

## Introduction

This repository is an official implementation of the <vSTMD: Visual Motion Detection for Extremely Tiny Target at Various Velocities>. The full work is under review for publication, and therefore only selected components are made available at this time.

## Sample Videos

<div style="display: flex; gap: 20px;">
  <div>
    <img src="https://github.com/mingshuoxu/vSTMD/raw/main/src/vSTMD_F_butterfly.gif" alt="flying butterfly" width="90%">
  </div>
  <div>
    <img src="https://github.com/mingshuoxu/vSTMD/raw/main/src/vSTMD_F_flying_bird.gif" alt="flying bird" width="90%">
  </div>
</div>

## Raw MP4 File Downloads

The following MP4 files are available for download in higher resolution compared to the GIF previews above. These videos provide clearer details for analysis and demonstration.

- [Download: Flying Butterfly (The first GIF above)](https://github.com/mingshuoxu/vSTMD/raw/main/src/vSTMD_F_butterfly.mp4)
- [Download: Flying Bird (The second GIF above)](https://github.com/mingshuoxu/vSTMD/raw/main/src/vSTMD_F_flying_bird.mp4)
- [Download: Result for RIST-GX010290-1](https://github.com/mingshuoxu/vSTMD/raw/main/src/vSTMD_F-GX010290-1.mp4)


## Pioneer Program

To explore the capabilities of vSTMD and vSTMD-F, you can download the repository from [GitHub: Small-Target-Motion-Detectors](https://github.com/MingshuoXu/Small-Target-Motion-Detectors). After downloading, you can run `start_by_python.py` using Python to get started.

## Current Status

The project is in the pre-publication phase. Some code and data have been omitted to comply with submission guidelines and ensure the integrity of the review process.

## Repository Contents

- **`comparison_models/`**: under update

- **`demo/`**: a demo for vSTMD and vSTMD-F.

- **`effective_of_direction/`**: Code for evaluating the effectiveness of direction.

- **`experience_in_RIST/`**: evaluation and ablation in RIST.

- **`groundtruth/`**: Some groundtruth in the panoramic datasets

- **`new_correlation_modelling/`**: for principle visualization

- **`parameter_analysis/`**: parameter analysis

- **`response_curve/`**: response curves

- **`result/`**: some evaluation results.

- **`src/`**: some example videos and gifs.
 
- **`velocity-AUC-curve/`**: module comparison between dynamic-and-correlation and delay-and-correlation.


## Limitations
- The full implementation, including parameter configurations and complete data processing pipelines, will be released after the publication of the related article.
- Some comments and detailed documentation may also be added in future updates.


## Future Plans
- Release the full codebase upon acceptance and publication of the article.
- Provide detailed documentation and extended examples.


## Contact
For questions or collaboration opportunities, feel free to contact me at [Mingshuoxu@hotmail.com](mailto:Mingshuoxu@hotmail.com).
