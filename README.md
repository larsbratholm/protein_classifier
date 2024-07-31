# Protein Classifier
Simple transformer encoder based classifier for amino acid sequences and relevant genes.
Assumes data in similar format to [dataset.csv](./tests/data/dataset.csv).

## Training models
Training can be done with the [train\_model.py](./scripts/train_model.py) script:
```
python train_model.py dataset.csv -m model_parameters.yaml -s trainer_parameters.yaml -o ./training_output -a gpu
```
The script automatically splits the data into training, validation and testing, but a separate test set can be given via the `-t` argument.
Example model and trainer parameters can be seen in e.g. [data/full](./data/full/).
If training on cpu, use `-a cpu`.

## Cross-validation
5-fold cross-validation can be done with the [cross\_validate.py](./scripts/cross_validate.py) script, which has the same input as above (except the `-t` argument).
The script trains 5 models and gathers the predictions and mean accuracy.

## Inference
Inference on data with an existing model can be run with the [inference.py](./scripts/inference.py) script:
```
python inference.py dataset.csv model.ckpt -o ./output -a gpu
```
Example checkpoint can be found in e.g. [data/full/split\_0/swa\_last.ckpt](./data/full/split_0/swa_last.ckpt).
