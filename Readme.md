# DMMR
This is the official PyTorch implementation for our AAAI'24 paper DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction  
[Paper link:](https://ojs.aaai.org/index.php/AAAI/article/view/27819)

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

## Citation
If you found our work useful for your research, please cite our work:
```
@inproceedings{wang2024dmmr,
  title={DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction},
  author={Wang, Yiming and Zhang, Bin and Tang, Yujiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={628--636},
  year={2024}
}
```
We thank the following repositories for providing helpful functions used in our work:
[MS-MDA](https://github.com/VoiceBeer/MS-MDA)  
[DANN](https://github.com/fungtion/DANN) 


