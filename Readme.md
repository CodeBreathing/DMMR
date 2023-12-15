# DMMR
Code of paper DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction

## Datasets
The public available datasets (SEED and SEED-IV) can be downloaded from the https://bcmi.sjtu.edu.cn/home/seed/index.html

To facilitate data retrieval, the data from the first session of all subjects is utilized in both datasets, the file structure of the datasets should be like:
```
ExtractedFeatures/
    1/
eeg_feature_smooth/
    1/
```
Kindly change the file path in the main.py

## Usage of DMMR
Run `python main.py`, and The results will be recorded in TensorBoard.
The argument for the `dataset_name` is set to be `seed3` for the SEED dataset, and `seed4` for the SEED-IV dataset, respectively.

## Ablation Studies
Run `python ablation/witoutMix.py`
Run `python ablation/withoutNoise.py`
Run `python ablation/withoutBothMixAndNoise.py`

## other noise injection methods
Run `python noiseInjectionMethods/maskChannels.py`
Run `python noiseInjectionMethods/maskTimeSteps.py`
Run `python noiseInjectionMethods/channelsShuffling.py`
Run `python noiseInjectionMethods/Dropout.py`

## Plot with TSNE
Run `python T-SNE/generatePlotByTSNE.py`

