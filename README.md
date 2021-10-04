# Detection Accuracy for Evaluating Compositional Explanations of Units

Find paper at: https://arxiv.org/abs/2109.07804

This repo extends the implementation of compositional explanations (vision experiments) at: https://github.com/jayelm/compexp/tree/master/vision

The code is built to generate NetDissect explanations, Compositional explanations (of length 3), and Compositional explanations whose lengths are determined using Detection Accuracy.

## How to run

1. Download Broden dataset by running the script `./script/dlbroden.sh` (only alexnet model requires downloading the 227 dataset)
2. Check the settings in `settings.py` and make sure they suite your desire.
3. You do not need to manually download the models; the code checks if the model to be probed as set in `settings.py` is available and automatically downloads it if not.
4. Run `python probe.py`. Results are generated in `result/` folder.

### Settings
All settings are in the `settings.py` file and are well commented.

To reduce execution time, consider reducing the size of candidates for beam search (`BEAM_SEARCH_LIMIT` parameter) and to probe only a subset of units by setting `UNIT_RANGE` parameter to a subset range or list of units.


### Results
All results are recorded in the `result/` folder. An additional folder is further created in this directory with the name structure `{model}_{probe_dataset}_neuron_{COMP_MAX_FORMULA_LENGTH}`. The output files are saved in this subfolder and are described below:

1. `tally_{layername}.csv`: records the explanations and the scores generated for the units set in `settings.py`.
2. `preds.csv`: records the predictions, the target labels, and the unit activations (boolean `1` or `0`) for the images in the dataset.
3. `correlation_scores.csv`: contains the scores of the correlation between the model accuracies and the evaluation scores.
4. `html/{layername}.html`: contains the html visualization of units for the layer probed (last layer).