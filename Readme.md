**vSTMD**: Visual Motion Detection for Extremely Tiny Target at Various Velocities
---

## Introduction

This repository is an official implementation of the <vSTMD: Visual Motion Detection for Extremely Tiny Target at Various Velocities>. The full work is under review for publication, and therefore only selected components are made available at this time.

## Sample Videos

<div style="display: flex; gap: 20px;">
  <div>
    <img src="https://github.com/mingshuoxu/vSTMD/raw/main/demo_videos/vSTMD_F_butterfly.gif" alt="flying butterfly" width="90%">
  </div>
  <div>
    <img src="https://github.com/mingshuoxu/vSTMD/raw/main/demo_videos/vSTMD_F_flying_bird.gif" alt="flying bird" width="90%">
  </div>
</div>

## Raw MP4 File Downloads

The following MP4 files are available for download in higher resolution compared to the GIF previews above. These videos provide clearer details for analysis and demonstration.

- [Download: Flying Butterfly (The first GIF above)](https://github.com/mingshuoxu/vSTMD/raw/main/example_result/vSTMD_F_butterfly.mp4)
- [Download: Flying Bird (The second GIF above)](https://github.com/mingshuoxu/vSTMD/raw/main/example_result/vSTMD_F_flying_bird.mp4)
- [Download: Result for RIST-GX010290-1](https://github.com/mingshuoxu/vSTMD/raw/main/example_result/vSTMD_F-GX010290-1.mp4)

## Application Scenarios
- Maritime Search and Rescue
  - YouTube Link:
    - [![demo1](https://img.youtube.com/vi/iS50CqMW3hg/0.jpg)](https://youtu.be/iS50CqMW3hg)
    - [![demo2](https://img.youtube.com/vi/yvSobdLzTRo/0.jpg)](https://youtu.be/yvSobdLzTRo)
    - [![demo3](https://img.youtube.com/vi/ga12Pt6Zucw/0.jpg)](https://youtu.be/ga12Pt6Zucw)
    - [![demo4](https://img.youtube.com/vi/9IWyRQUEtGU/0.jpg)](https://youtu.be/9IWyRQUEtGU)
    - [![demo5](https://img.youtube.com/vi/xqy9B-ARLPU/0.jpg)](https://youtu.be/xqy9B-ARLPU)

  - Dataset link:
    - SeaDronesSee: https://seadronessee.cs.uni-tuebingen.de/home

## Pioneer Program

To explore the capabilities of vSTMD and vSTMD-F, you can download the repository from [GitHub: Small-Target-Motion-Detectors](https://github.com/MingshuoXu/Small-Target-Motion-Detectors). After downloading, you can run `start_by_python.py` using Python to get started.

## Current Status

The project is in the pre-publication phase. Some code and data have been omitted to comply with submission guidelines and ensure the integrity of the review process.

## Repository Contents

- **`comparison_models/`**: 
  - `custom_API.py`: custom API for comparison models.
  - `LC_neuron.py`: LC11 and LC18 neuron models.

- **`demo/`**: a demo for vSTMD and vSTMD-F.

- **`demo_videos/`**: some example videos and gifs.

- **`effective_of_direction/`**: Code for evaluating the effectiveness of direction (Sect. V-B2 in the main text).
  - `show_result.ipynb`: Tab II in the main text, and Tab. III in the supplementary material.

- **`evaluate_result/`**: some results.
  - `RIST/`: results for RIST dataset.
  - `vSTMD_Panorama_Stimuli/`: results for panoramic stimuli.
  - `XS-VID/`: results for XS-VID dataset.

- **`experience_in_RIST/`**: experiments in RIST.
  - `ablation/`: ablation studies (Sect. V-D in the main text).
    - `show_ablation_result.ipynb`: Tab. IV in the main text.
  - `comparison/`: comparison with other models (Sect. V-C in the main text).
    - `show_result_in_RIST.ipynb`: Tab. III in the main text.
  - `parameter_analysis/`: parameter analysis.
    - `parameter_analysis.py`: Fig. 4 in the supplementary material.
  - `statistics_mean_para/`: statistical results for mean parameter settings.
    - `velocity_range.py`: Fig. 1-3 in the supplementary material.
  - `visulize_for_RIST/`: visualization for RIST dataset. (Sect. V-C in the main text).
    - `visulize_by_plt.py`: Fig. 8 in the main text.

- **`experience_in_XS-VID/`**: experiments in XS-VID dataset (Sect. VI-C2 in the main text).
  - `show_result_in_XS-VID.ipynb`: Tab. VI in the main text.

- **`groundtruth/`**: Some groundtruth in the panoramic datasets (Sect. V-B in the main text).

- **`maritime/`**: maritime search and rescue related code. (Sect. VI-C1 in the main text).
  - `inference_model.py`: Tab. I and II in the supplementary material.
  - `model_demo.py`: maritime demo for vSTMD and vSTMD-F. Fig. 6 in the supplementary material.

- **`modelling_plot/`**: for principle visualization (Sect. IV in the main text).
  - `various_velocity_effectiveness.py`: Fig. 3 in the main text.
  - `leaky_integrate_model.m`: Fig. 4 in the main text.
  - `direction_modelling.m`: Fig. 6 in the main text.

- **`response_curve/`**: response curves
  - `size_and_contrast_cruve.py`: Fig. 5 in the supplementary material.

- **`timecost_analysis/`**: time cost analysis (Sect. V-E in the main text).
  - `module_analysis.py`: Tab. V in the main text.
  - `CPU_vs_GPU_analysis.py`: Tab III　in the main text.
 
- **`velocity-AUC-curve/`**: module comparison between dynamic-and-correlation and delay-and-correlation. (Sect. V-B1 in the main text).
  - visualize_velocity_AUC_curve.py: Fig. 7 in the main text.

## Limitations
- The full implementation, including parameter configurations and complete data processing pipelines, will be released after the publication of the related article.
- Some comments and detailed documentation may also be added in future updates.


## Future Plans
- Release the full codebase upon acceptance and publication of the article.
- Provide detailed documentation and extended examples.


## Contact
For questions or collaboration opportunities, feel free to contact me at [Mingshuoxu@hotmail.com](mailto:Mingshuoxu@hotmail.com).
