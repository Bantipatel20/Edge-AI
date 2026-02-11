# Drug Powder Mixing Classification Project

## Overview
This project classifies pharmaceutical powder images as either **Mixed** (properly homogenized) or **Not Mixed** (improperly mixed) using deep learning. It uses a DenseNet121 model adapted for 9-channel input (RGB + HSV + Lab) and provides a full pipeline from dataset splitting to model training, evaluation, and metrics reporting.

## Folder Structure
```
Final_Code/
  main_training_pipeline.py   # Main script for splitting, training, and evaluation
  generate_gradcam_overlays.py # Batch Grad-CAM overlays
  generate_gradcam_grid.py     # Grid of original vs Grad-CAM overlays
  extract_layer_feature_maps.py # Feature map extraction
  metrics_and_plots.py         # Confusion matrix, ROC, PR, metrics plots
  model_comparison.py          # Model comparison bar chart
  class_distribution_and_error_analysis.py # Class distribution & error analysis
  requirements.txt             # Environment setup
  README.md                    # Project documentation
  GradCAM_Overlays/            # Grad-CAM overlays for test images
  plots/                       # Grids, metrics charts, ROC curves, etc.
  layer_outputs/               # Feature maps from model layers
```

## Dataset Structure
Your dataset should be organized as:
```
YourDatasetFolder/
    Mixed/         # Images of properly mixed powders
    Not Mixed/     # Images of improperly mixed powders
```
All images should be 512x512 pixels.

## Steps to Run
1. **Edit Paths**
   - In `main_training_pipeline.py`, set:
     - `dataset_dir` to your dataset folder
     - `output_dir` to where you want the train/val/test splits

2. **Install Requirements**
   - Run:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Main Pipeline**
   - Execute `main_training_pipeline.py`:
     ```
     python main_training_pipeline.py
     ```
   - The script will:
     - Split your dataset into train/val/test
     - Train a DenseNet121 model on 9-channel images
     - Evaluate and print metrics for all splits
     - Save the best model as `best_densenet_9ch.pth`

4. **Generate Grad-CAM Overlays**
   - Run `generate_gradcam_overlays.py` to create overlays for test images:
     ```
     python generate_gradcam_overlays.py
     ```
   - Overlays are saved in `GradCAM_Overlays/`

5. **Create Grad-CAM Grids**
   - Run `generate_gradcam_grid.py` to create visual grids:
     ```
     python generate_gradcam_grid.py
     ```
   - Grids are saved in `plots/`

6. **Extract Feature Maps**
   - Run `extract_layer_feature_maps.py` for layer visualizations:
     ```
     python extract_layer_feature_maps.py
     ```
   - Feature maps are saved in `layer_outputs/`

7. **Plot Metrics and Model Comparison**
   - Use `metrics_and_plots.py` and `model_comparison.py` to generate metrics charts and compare models.
   - Use `class_distribution_and_error_analysis.py` for class distribution and error analysis plots.

## Model Details
- **Architecture**: DenseNet121 (pretrained, modified for 9 channels)
- **Input**: 9-channel images (RGB + HSV + Lab)
- **Classes**: Mixed (0), Not Mixed (1)
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC, PR curve

## Customization
- Change batch size, epochs, or learning rate in the script as needed.
- You can add Grad-CAM visualization or more advanced models if desired.
- You can add more models or visualizations as desired.

## Requirements
See `requirements.txt` for all dependencies.

## Reproducibility
- All random seeds are set for reproducible splits and training.
- Use `requirements.txt` for environment setup.

## Contact
For questions or improvements, contact the project owner.
