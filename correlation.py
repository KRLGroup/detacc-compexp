import os
import numpy as np
from loader.data_loader import ade20k
from loader.data_loader.broden import SegmentationData
from loader.data_loader.catalog import get_mask_global
import settings
from tqdm import tqdm
# from evaluation import utils
from pycocotools import mask as cmask
from PIL import Image
from util import misc


def get_formula_unit_img_idx(masks, ufeat, uthresh, data_mask_shape, u_img_acts, comb_type="and"):
    """
    Fetches indexes of imgs in dataset where unit fires and/or explanation exists based on 'comb_type'
    Args:
        masks: (MSCOCO data) the mask of the explanation under consideration over the dataset
        ufeat: (ndarray) features of the unit over the dataset
        uthresh: (float) the activation threshold for this unit
        data_mask_shape: (tuple) new dimension to upsample the features to
        u_img_acts: (ndarray - 1D) precomputed idexes of imgs where unit fired beyond threshold
        comb_type: (string) value indicating how to combine unit activations and explanation presence on imgs. Values are 'and' | 'or'
    Returns:
        formula_imgs: (ndarray - 1D) linear array of indexes of imgs where unit activates and/or explanation is present
    """
    masks_np = cmask.decode(masks)
    if len(masks_np.shape) == 2:
        masks_np = masks_np.reshape(data_mask_shape)
    else:
        masks_np = np.einsum('jki->ijk', masks_np) # transpose
    
    formula_imgs = np.any(masks_np, axis=(1,2))
    
    if comb_type=="and":
        uall_uhitidx = misc.get_unit_acts(
            ufeat, uthresh, (data_mask_shape[1], data_mask_shape[2]), data_mask_shape[0]
        )
        overlap = np.logical_and(masks_np, uall_uhitidx)
        formula_imgs = np.any(overlap, axis=(1,2))
    else:
        formula_imgs = np.logical_or(formula_imgs, u_img_acts)
    
    formula_imgs = np.argwhere(formula_imgs).ravel()
    
    return formula_imgs


def get_accuracy_list(preds, img_idxs, data):
    """
    given an image set and the corresponding model predictions, computes an accuracy list
    Args:
        preds: list of all model predictions over dataset
        img_idxs: a list of subset of img indexes
        data: object of SegmentationData class located in 'loader/data_loader/broden'
    
    Returns:
        acc_list: list containing the truth value of model predictions for each img in the subset provided, ie., for each img in the subset, true is recorded if the prediction is correct and false otherwise
    """
    acc_list = []
    u_preds = preds[img_idxs]
    for i, (p, t) in enumerate(u_preds):
        pred_name = ade20k.I2S[p]
        target_name = f"{data.scene(img_idxs[i])}-s"
        if target_name in ade20k.S2I:
            acc_list.append(pred_name==target_name)
    
    return acc_list


def compute(records, mc, preds, allacts, feats, threshs):
    """
    computes the correlation of the evaluation scores (IoU and Detection Accuracy), and writes the output to file 'correlation_scores.csv' in the folder set in settings.OUTPUT_FOLDER
    Args:
        records: (list) list of dictionaries, each containing the explanations and scores for a unit
        mc: object of the MaskCatalog class located in 'loader/data_loader/catalog'
        preds: list of all model predictions over dataset
        allacts: (ndarray - 1D) array of boolean values over all over dataset, true for imgs on which unit activates beyond the threshold
        feats: (ndarray - 4D) array of all features of a layer of the model, for all units and over all imgs
        threshs: (ndarray - 1D) array of threshold values for each unit of the model

    Returns:
        None
    """
    data = SegmentationData(settings.DATA_DIRECTORY, categories=settings.CATEGORIES)
    data_mask_shape = (data.size(), *mc.mask_shape)
    
    scores_lists = {
        "detacc_expl": {"detacc": [], "iou": [], "and_acc": [], "or_acc": []}, 
        "comp_expl": {"detacc": [], "iou": [], "and_acc": [], "or_acc": []}, 
        "netdissect": {"detacc": [], "iou": [], "and_acc": [], "or_acc": []}
    }
    for unit_data in tqdm(records, desc=f"Computing correlation scores"):
        u = unit_data["unit"]
        titles = ["detacc_expl", "comp_expl", "netdissect"]
        formulas = [unit_data["formula"], unit_data["comp_form"], unit_data["netdissect_form"]]
        detacc_scores = [unit_data["detacc"], unit_data["comp_detacc"], unit_data["netdissect_detacc"]]
        iou_scores = [unit_data["iou"], unit_data["comp_iou"], unit_data["netdissect_iou"]]

        for title, formula, detacc, iou in zip(titles, formulas, detacc_scores, iou_scores):
            masks = get_mask_global(mc.masks, formula)
            
            scores_lists[title]["detacc"].append(detacc)
            scores_lists[title]["iou"].append(iou)

            for comb_type in ["and", "or"]:
                img_idxs = get_formula_unit_img_idx(masks, feats[:, u], threshs[u], data_mask_shape, allacts[:, u], comb_type)
                acc_list = get_accuracy_list(preds, img_idxs, data)
                if acc_list:
                    accuracy = np.mean(acc_list)
                    scores_lists[title][comb_type+"_acc"].append(accuracy)
                else:
                    scores_lists[title][comb_type+"_acc"].append(0.5) # average of all it could be

    savepath = os.path.join(settings.OUTPUT_FOLDER, "correlation_scores.csv")

    output = """,DetAcc,,IoU
,OR,AND,OR,AND
DetAcc Expl:,{},{},{},{}
Comp Expl:,{},{},{},{}
NetDissect:,{},{},{},{}
""".format(
        round(np.corrcoef(scores_lists["detacc_expl"]["detacc"], scores_lists["detacc_expl"]["or_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["detacc_expl"]["detacc"], scores_lists["detacc_expl"]["and_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["detacc_expl"]["iou"], scores_lists["detacc_expl"]["or_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["detacc_expl"]["iou"], scores_lists["detacc_expl"]["and_acc"])[0, 1], 4),

        round(np.corrcoef(scores_lists["comp_expl"]["detacc"], scores_lists["comp_expl"]["or_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["comp_expl"]["detacc"], scores_lists["comp_expl"]["and_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["comp_expl"]["iou"], scores_lists["comp_expl"]["or_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["comp_expl"]["iou"], scores_lists["comp_expl"]["and_acc"])[0, 1], 4),

        round(np.corrcoef(scores_lists["netdissect"]["detacc"], scores_lists["netdissect"]["or_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["netdissect"]["detacc"], scores_lists["netdissect"]["and_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["netdissect"]["iou"], scores_lists["netdissect"]["or_acc"])[0, 1], 4),
        round(np.corrcoef(scores_lists["netdissect"]["iou"], scores_lists["netdissect"]["and_acc"])[0, 1], 4),
    )
    
    with open(savepath, "w") as f:
        f.write(output)

