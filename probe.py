"""
Reference: https://github.com/jayelm/compexp/blob/master/vision/probe.py

"""

import settings 
from loader.model_loader import loadmodel 
from dissection.neuron import hook_feature, NeuronOperator
from visualize.report import generate_html_summary
from util.misc import safe_layername
import os
import pandas as pd
from loader.data_loader import ade20k
import numpy as np

import correlation

import urllib.request as req
def handle_model_download():
    if not os.path.exists(settings.MODEL_FILE):
        
        print(f"Model {settings.MODEL} not found. Downloading it...")
        req.urlretrieve(
            settings.MODEL_URLS[settings.MODEL], 
            settings.MODEL_FILE
        )
        print(f"Download complete. Execution continues...")


layernames = list(map(safe_layername, settings.FEATURE_NAMES))
hook_modules = []

if __name__ == "__main__":
    # First check if the model to probe as set in settings.py is available. If not, download it.
    handle_model_download()

    # Load model
    model = loadmodel(hook_feature, hook_modules=hook_modules)
    fo = NeuronOperator()

    # ==== STEP 1: Feature extraction ====
    # features: list of activations - one 63305 x c x h x w tensor for each feature
    # layer (defined by settings.FEATURE_NAMES; default is just layer4)
    # maxfeature: the maximum activation across the input map for each channel (e.g. for layer 4, there is a 7x7 input map; what's the max value). one 63305 x c tensor for each feature
    features, maxfeature, preds, logits = fo.feature_extraction(model=model)
    print(f"\nFeatures shape: {features[-1].shape}\n")
    
    # ==== STEP 2: Threshold quantization ====
    thresholds = [
        fo.quantile_threshold(lf, savepath=f"quantile_{ln}.npy")
        for lf, ln in zip(features, layernames)
    ]

    # ==== Average feature activation quantiling ====
    # You can threshold by quantiling top alpha% of MEAN activations...
    #  wholeacts = fo.get_total_activation(features[-1])
    # ...or simply wherever there's a hit
    wholeacts = features[-1] > thresholds[-1][np.newaxis, :, np.newaxis, np.newaxis]
    wholeacts = wholeacts.any((2, 3))

    # ==== Confusion matrix =====
    pred_records = []
    for i, ((p, t), acts) in enumerate(zip(preds, wholeacts)):
        acts = acts * 1  # To int
        pred_name = ade20k.I2S[p]
        target_name = f"{fo.data.scene(i)}-s"
        if target_name in ade20k.S2I:
            pred_records.append((pred_name, target_name, *acts))

    pred_df = pd.DataFrame.from_records(
        pred_records, columns=["pred", "target", *map(str, range(wholeacts.shape[1]))]
    )
    pred_df.to_csv(os.path.join(settings.OUTPUT_FOLDER, "preds.csv"), index=False)
    print(f"Accuracy: {(pred_df.pred == pred_df.target).mean() * 100:.2f}%")

    # ==== STEP 3: Generating explanations ====
    if settings.UNIT_RANGE is None:
        tally_dfname = f"tally_{layernames[-1]}.csv"
    else:
        # If only a subset of units is used
        tally_dfname = f"tally_{layernames[-1]}_{min(settings.UNIT_RANGE)}_{max(settings.UNIT_RANGE)}.csv"

    tally_result, mc = fo.tally(
        features[-1], thresholds[-1], savepath=tally_dfname
    )
    
    correlation.compute(
        tally_result, mc, preds, wholeacts, features[-1], thresholds[-1]
    )

    # ==== STEP 4: generating results ====
    generate_html_summary(fo.data, layernames[-1], mc.mask_shape,
            tally_result=tally_result, thresholds=thresholds[-1],
            maxfeature=maxfeature[-1], features=features[-1],
    )

