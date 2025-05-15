# REFINED-BIAS: A Disentangled Benchmark and Framework for Integrated Evaluation of Shape/Texture Bias in Neural Networks


## Abstract
Mixed visual cue images, such as cue-conflict images, are known to effectively
reveal how deep neural networks perceive cues like texture and shape. However, we argue that the influential cue-conflict benchmark suffers from three key limitations: (1) stylization introduces artifacts that blur the distinction between texture and shape cues; (2) evaluation is limited by a small class subset, distorting the full output distribution of n-way models (e.g., 1,000-way for ImageNet pre-trained models); and (3) evaluation metrics are skewed by dominant classes particularly when the model exhibits low precision. These issues distort bias evaluation and weaken alignment with human perception. To address this, we introduce REFINED-BIAS, a diagnostic benchmark that corrects these distortions and better aligns with human visual understanding. REFINED-BIAS generates artifact-free samples using human-recognizable patterns and quantifies cue sensitivity across the full label space using Mean Reciprocal Rank, enabling more robust and interpretable evaluations. Our study show that REFINED-BIAS provides more accurate and human-aligned insights than the prior benchmark. Applied across diverse architectures and training regimes, REFINED-BIAS offers a corrected view on texture and shape biases, setting a new standard for probing visual priors in neural networks. Our code is available at REFINED-BIAS .

## How to reproduce our results
### Environmental Setup <!-- 수정해야 됨 -->
```
conda create --name refined-bias python=3.8.20 -y
conda activate refined-bias
pip install -r ./requirements.txt
```

### Dataset
You can evaluate our REFINED-BIAS shape cue & Texture cue in `./datasets/refined_bias_shape` and `./datasets/refined_bias_texture`. The dataset structure is shown as:
```
datasets
├── refined_bias_shape
│   ├── balloon
│       ├── balloon_0.png
│       ├── balloon_1.png
│           ...
│   ├── book
│   ...
└── refined_bias_texture
│   ├── brain_coral
│       ├── 4x4_brain_coral_0.png
│       ├── 4x4_brain_coral_1.png
│           ...
│   ├── texutre
    ...
* shape : (3, 224, 224) 
```
### Evaluation
To evaluate REFINED-BIAS across different model architectures and learning strategies, simply run the following command. The necessary checkpoints will be fetched and downloaded automatically.
```
# evaluate across model architecture on REFINED-BIAS shape cue
python eval_refined_bias.py --dataset refined_bias_shape --across arch

# evaluate across model architecture on REFINED-BIAS texture cue
python eval_refined_bias.py --dataset refined_bias_texture --across arch

# evaluate across learning strategy on REFINED-BIAS shape cue
python eval_refined_bias.py --dataset refined_bias_shape --across strategy

# evaluate across learning strategy on REFINED-BIAS texture cue
python eval_refined_bias.py --dataset refined_bias_texture --across strategy

```
The above command will print out as shown in the example below:
```
REFINED-BIAS Shape Bias on arch
> bagnet9               : 0.0518
> bagnet17              : 0.0988
> bagnet33              : 0.2438
> ...
```
Detailed per-class scores for each model and learning strategy can be found in the .json files located under `./results/across_model_architecture` and `./results/across_learning_strategy`.
