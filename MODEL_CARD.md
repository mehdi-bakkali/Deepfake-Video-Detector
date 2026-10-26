# Model card

## Summary

This repository provides a baseline video classifier. It fine-tunes ImageNet-pretrained EfficientNet-B0 on multiple frames per video from either the Google Deepfake Detection (DFD) dataset or the DeepFake Detection Challenge (DFDC). Frame logits are averaged to produce a video-level fake probability.

## Intended use

- Learning and experimentation with video deepfake classification.
- Reproducible baseline comparisons on data that the user is permitted to access.
- Human-reviewed research demonstrations.

## Out-of-scope use

- Establishing the authenticity of a video or a person's identity.
- Moderation, employment, law enforcement, financial, or other consequential decisions without independent validation and human review.
- Treating the score as a calibrated probability or forensic conclusion.

## Training data

The loader accepts either DFDC `metadata.json` files with `REAL` or `FAKE` labels or the DFD folder layout, where parent folder names identify original and manipulated videos. Data is not distributed here; users must obtain it from Kaggle and follow the applicable Kaggle and upstream source terms. The split groups a fake with its source original when metadata provides that relationship. Folder-labeled DFD records do not expose a source-pair mapping through the folder names alone, so its split is stratified by video rather than guaranteed subject-independent.

## Model and inputs

- Backbone: Torchvision EfficientNet-B0 initialized with ImageNet weights.
- Task: binary frame classification; `REAL` maps to 0 and `FAKE` maps to 1.
- Training input: multiple randomly sampled frames per video per epoch; validation and inference use evenly spaced frames.
- Preprocessing: optionally crop around the largest OpenCV frontal-face detection, falling back to the full frame; resize and center-crop to 224 × 224.
- Video aggregation: mean frame logits, used consistently in training, validation, and inference.
- Audio is not analyzed. Faces are not tracked or aligned.

## Evaluation

No trained checkpoint or benchmark result is included. The training command reports validation loss, ROC-AUC, average precision, and balanced accuracy on its grouped holdout split. Users should evaluate on a held-out dataset that reflects the intended setting, report class balance and operating thresholds, and avoid selecting a threshold on the final test set. A small sample is useful for a workflow check, not a reliable estimate of generalization.

## Limitations and risks

Performance may vary with manipulation method, face size and position, lighting, compression, resolution, subject demographics, and other dataset differences. The simple face detector may miss profiles or unusual lighting, in which case the full frame is used. Averaging frames can hide localized manipulations and does not provide calibrated uncertainty. False positives and false negatives are expected, particularly on data outside the training distribution.

## Responsible use

Use the output as one investigative signal alongside provenance, source context, and human review. Do not present a prediction as proof that a person created or appeared in manipulated media. Respect the dataset's access and use terms and avoid publishing private or sensitive videos.
