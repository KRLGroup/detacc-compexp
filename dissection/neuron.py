import os
from PIL import Image
import numpy as np
import torch
import settings
# import util.upsample as upsample
import pandas as pd
import util.vecquantile as vecquantile
from util.misc import safe_layername
import multiprocessing.pool as pool
import multiprocessing as mp
from loader.data_loader.broden import load_csv
from loader.data_loader.broden import SegmentationData, SegmentationPrefetcher
from loader.data_loader.catalog import MaskCatalog, get_mask_global
from loader.data_loader import formula as F
from tqdm import tqdm, trange
import csv, copy
from collections import Counter
import itertools
import pickle

from pycocotools import mask as cmask

features_blobs = []

def hook_feature(module, inp, output):
    features_blobs.append(output.data.cpu().numpy())


def upsample_features(features, shape):
    return np.array(Image.fromarray(features).resize(shape, resample=Image.BILINEAR))


def get_last_unit(records):
    """
    ADAPTIVE RESTART: In case execution breaks, this function gets the last unit finished so execution can pick up from there
    """
    units = [-1]
    for rec in records:
        units.append(int(rec["unit"]))
    return np.sort(units)[-1]


class NeuronOperator:
    def __init__(self):
        os.makedirs(os.path.join(settings.OUTPUT_FOLDER, "image"), exist_ok=True)
        self.data = SegmentationData(
            settings.DATA_DIRECTORY, categories=settings.CATEGORIES
        )
        self.loader = SegmentationPrefetcher(
            self.data,
            categories=["image"],
            once=True,
            batch_size=settings.BATCH_SIZE,
        )
        self.mean = [109.5388, 118.6897, 124.6901]

    def feature_extraction(
        self,
        model=None,
        memmap=True,
        feature_names=settings.FEATURE_NAMES,
        features_only=False,
    ):
        loader = self.loader
        # extract the max value activaiton for each image
        maxfeatures = [None] * len(feature_names)
        wholefeatures = [None] * len(feature_names)
        all_preds = None
        all_logits = None
        features_size = [None] * len(feature_names)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")

        if memmap:
            skip = True
            mmap_files = [
                os.path.join(
                    settings.OUTPUT_FOLDER, "%s.mmap" % safe_layername(feature_name)
                )
                for feature_name in feature_names
            ]
            mmap_max_files = [
                os.path.join(
                    settings.OUTPUT_FOLDER, "%s_max.mmap" % safe_layername(feature_name)
                )
                for feature_name in feature_names
            ]
            mmap_pred_file = os.path.join(settings.OUTPUT_FOLDER, "pred.mmap")
            mmap_logit_file = os.path.join(settings.OUTPUT_FOLDER, "logit.mmap")
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, (mmap_file, mmap_max_file) in enumerate(
                zip(mmap_files, mmap_max_files)
            ):
                if (
                    os.path.exists(mmap_file)
                    and os.path.exists(mmap_max_file)
                    and os.path.exists(mmap_pred_file)
                    and os.path.exists(mmap_logit_file)
                    and features_size[i] is not None
                ):
                    print("loading features %s" % safe_layername(feature_names[i]))
                    wholefeatures[i] = np.memmap(
                        mmap_file,
                        dtype=np.float32,
                        mode="r",
                        shape=tuple(features_size[i]),
                    )
                    maxfeatures[i] = np.memmap(
                        mmap_max_file,
                        dtype=np.float32,
                        mode="r",
                        shape=tuple(features_size[i][:2]),
                    )
                else:
                    print("file missing, loading from scratch")
                    skip = False
            # Single logit/pred files
            if os.path.exists(mmap_pred_file) and os.path.exists(mmap_logit_file):
                all_preds = np.memmap(
                    mmap_pred_file,
                    dtype=np.int64,
                    mode="r",
                    shape=(features_size[i][0], 2),
                )
                all_logits = np.memmap(
                    mmap_logit_file,
                    dtype=np.float32,
                    mode="r",
                    shape=(features_size[i][0], settings.NUM_CLASSES),
                )
            else:
                skip = False
            # Single logit/pred files
            if skip:
                return wholefeatures, maxfeatures, all_preds, all_logits

        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
        for batch_idx, (inp, targets, *_) in tqdm(
            enumerate(loader.tensor_batches(bgr_mean=self.mean, global_labels=True)),
            desc="Extracting features",
            total=int(np.ceil(num_batches)),
        ):
            del features_blobs[:]
            inp = torch.from_numpy(inp[:, ::-1, :, :].copy())
            inp.div_(255.0 * 0.224)
            if settings.GPU:
                inp = inp.cuda()
            with torch.no_grad():
                logits = model.forward(inp)

            while np.isnan(logits.data.cpu().max()):
                print("nan")
                del features_blobs[:]
                logits = model.forward(inp)

            preds = logits.argmax(1).cpu()
            targets = targets.squeeze(1)

            if not features_only:
                if all_preds is None:
                    size_preds = (len(loader.indexes), 2)
                    if memmap:
                        all_preds = np.memmap(
                            mmap_pred_file, dtype=np.int64, mode="w+", shape=size_preds
                        )
                    else:
                        all_preds = np.zeros(size_preds, dtype=np.int64)

                if all_logits is None:
                    size_logits = (len(loader.indexes), settings.NUM_CLASSES)
                    if memmap:
                        all_logits = np.memmap(
                            mmap_logit_file,
                            dtype=np.float32,
                            mode="w+",
                            shape=size_logits,
                        )
                    else:
                        all_logits = np.zeros(size_logits, dtype=np.float32)

            if maxfeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (len(loader.indexes), feat_batch.shape[1])
                    if memmap:
                        maxfeatures[i] = np.memmap(
                            mmap_max_files[i],
                            dtype=np.float32,
                            mode="w+",
                            shape=size_features,
                        )
                    else:
                        maxfeatures[i] = np.zeros(size_features)

            if len(feat_batch.shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (
                        len(loader.indexes),
                        feat_batch.shape[1],
                        feat_batch.shape[2],
                        feat_batch.shape[3],
                    )
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(
                            mmap_files[i],
                            dtype=np.float32,
                            mode="w+",
                            shape=size_features,
                        )
                    else:
                        wholefeatures[i] = np.zeros(size_features)

            np.save(features_size_file, features_size)
            start_idx = batch_idx * settings.BATCH_SIZE
            end_idx = min((batch_idx + 1) * settings.BATCH_SIZE, len(loader.indexes))
            for i, feat_batch in enumerate(features_blobs):
                if len(feat_batch.shape) == 4:
                    wholefeatures[i][start_idx:end_idx] = feat_batch
                    maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch, 3), 2)
                elif len(feat_batch.shape) == 3:
                    maxfeatures[i][start_idx:end_idx] = np.max(feat_batch, 2)
                elif len(feat_batch.shape) == 2:
                    maxfeatures[i][start_idx:end_idx] = feat_batch

            if not features_only:
                all_preds[start_idx:end_idx] = np.stack((preds, targets), 1)
                all_logits[start_idx:end_idx] = logits.cpu().numpy()

        if len(feat_batch.shape) == 2:
            wholefeatures = maxfeatures

        return wholefeatures, maxfeatures, all_preds, all_logits

    def quantile_threshold(self, features, savepath=""):
        """
        Determine thresholds for neuron activations for each neuron.
        """
        qtpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(qtpath):
            print(f"Loading cached quantiles {qtpath}")
            return np.load(qtpath)
        print("calculating quantile threshold")
        quant = vecquantile.QuantileVector(depth=features.shape[1], seed=1)
        batch_size = 64
        for i in trange(0, features.shape[0], batch_size, desc="Processing quantiles"):
            batch = features[i : i + batch_size]
            batch = np.transpose(batch, axes=(0, 2, 3, 1)).reshape(
                -1, features.shape[1]
            )
            quant.add(batch)
        ret = quant.readout(1000)[:, int(1000 * (1 - settings.QUANTILE) - 1)]
        if savepath:
            np.save(qtpath, ret)
        return ret

    def get_total_activation(self, features):
        """
        Input:
        data_size x n_neurons x (mask_dim)

        Average representations over all, then quantile and keep top percent
        """
        mfeat = features.mean((2, 3))
        thresh = np.quantile(mfeat, 1 - settings.TOTAL_QUANTILE, axis=1, keepdims=True)
        hits = mfeat > thresh
        return hits

    @staticmethod
    def get_uhits(args):
        """
            Given neuron activations, upsample to the mask resolution, then
            threshold to create binary masks which are then stored in compressed
            MSCOCO mask form.

            Returns:
                u: the unit
                uidx: all image indices where the unit exceeds the threshold
                uhit_mask: the compressed MSCOCO unit mask across all datapoints
                uhits: the total number of times (pixels) the neuron fires
                uhitidx_img: boolean values representing the activation on each datapoint
                uhits_img: the total number of times (images) the neuron fires
        """
        (u, ufeat_, uthresh, mask_shape, data_size) = args
        
        uidx = np.argwhere(ufeat_.max((1, 2)) > uthresh).squeeze(1)
        ufeat = np.array([upsample_features(ufeat_[i], mask_shape) for i in uidx])
        if ufeat.shape[0] < 1:
            uidx = np.array([0])
            ufeat = np.array([upsample_features(ufeat_[i], mask_shape) for i in uidx])

        # Create full array
        uhitidx = np.zeros((data_size, *mask_shape), dtype=np.bool)
        uhitidx_img = np.zeros((data_size,), dtype=np.bool)

        # Get indices where threshold is exceeded
        uhit_subset = ufeat > uthresh
        uhits = uhit_subset.sum()

        uhit_subset_img = np.any(uhit_subset, axis=(1,2))
        uhits_img = uhit_subset_img.sum()

        if uhits > 0:
            uhitidx[uidx] = uhit_subset
            uhitidx_img[uidx] = uhit_subset_img

        # Save as compressed
        uhitidx_flat = uhitidx.reshape(
            (uhitidx.shape[0] * uhitidx.shape[1], uhitidx.shape[2])
        )
        uhit_mask = cmask.encode(np.asfortranarray(uhitidx_flat))

        return u, uidx, uhit_mask, uhits, uhitidx_img, uhits_img

    @staticmethod
    def _tally(features, data, threshold, start, end, savepath, csvpath):
        """
            Internal tally method

            Params:
                features: neuron activations across the entire dataset (shape: n_images x n_neurons x *activation_map)
                data: SegmentationData
                threshold: thresholds for each neuron shape: (n_neurons)
                start/end: only compute for a subset of the data

            Computes binary masks for neurons, measures IoU w/ compositional
            search, and saves resulting formulas/ious to `csvpath`.
        """
        g = {}
        namer = lambda name: 'NONE' if name is None else data.name(None, name)
        cat_namer = lambda name: 'NONE' if name is None else categories[pcats[name]]

        categories = data.category_names()
        pcpi = data.primary_categories_per_index()
        g["pcpi"] = pcpi
        pcats = data.primary_categories_per_index()
        
        if settings.UNIT_RANGE is not None:
            units = len(settings.UNIT_RANGE)
            units_ = settings.UNIT_RANGE
        else:
            units = features.shape[1]
            units_ = range(features.shape[1])

        # Segmentation loader
        pf = SegmentationPrefetcher(
            data,
            categories=categories,
            once=True,
            batch_size=settings.TALLY_BATCH_SIZE,
            ahead=settings.TALLY_AHEAD,
            start=start,
            end=end,
        )
        # Cache all masks so they can be looked up
        mc = MaskCatalog(pf)

        # Setting mc, labels, masks, img2cat, mask_shape, data_mask_shape for g
        g["mc"] = mc
        g["labels"] = mc.labels
        g["masks"] = mc.masks
        g["img2cat"] = mc.img2cat
        g["mask_shape"] = mc.mask_shape
        g["data_mask_shape"] = (features.shape[0], *mc.mask_shape)

        # Cache label tallies
        # TALLY_LABELS: number of times a label hits across the dataset
        g["tally_labels"] = {}
        g["tally_labels_img"] = {}
        
        # COUNTING FEATURES
        count = 0
        for lab in tqdm(mc.labels, desc="Tally labels"):
            masks = g["masks"][lab] #masks = mc.get_mask(F.Leaf(lab))

            masks_np = cmask.decode(masks)
            masks_np = masks_np.reshape(g["data_mask_shape"])
            img_has_lab = np.any(masks_np, axis=(1,2))

            # Counting all images that have this concept in them
            label_occurence = np.einsum('i->', img_has_lab, dtype=int)
            
            # Skip concepts that don't appear in at least CONCEPT_MIN_OCCURENCE images
            if label_occurence < settings.CONCEPT_MIN_OCCURENCE:
                continue

            g["tally_labels_img"][lab] = label_occurence
            g["tally_labels"][lab] = cmask.area(masks)

            # Catering to 'NOT' labels
            n_masks = mc.get_mask(F.Not(F.Leaf(lab)))
            g["masks"][-lab] = n_masks
            n_masks_np = cmask.decode(n_masks)
            n_masks_np = n_masks_np.reshape(g["data_mask_shape"])
            img_has_n_lab = np.any(n_masks_np, axis=(1,2))
            g["tally_labels_img"][-lab] = sum(img_has_n_lab)
            

            if g["tally_labels"][lab] > 0:
                count += 1 
        
        mc.labels = g["tally_labels"].keys()
        print(f"# nonzero concepts: {count}")

        tally_units = {u: 0 for u in units_}
        
        g["tally_units"] = tally_units
        g["tally_units_img"] = {u:0 for u in units_}

        # Get unit information (this is expensive (upsampling) and where most of the work is done)
        g["features"] = features
        g["threshold"] = threshold
        
        mp_args = (
            (
                u, 
                g["features"][:, u],
                g["threshold"][u],
                g["mask_shape"],
                g["features"].shape[0]
            )
            for u in units_
        )

        all_uidx = {u: None for u in units_}
        all_uhitidx = {u: None for u in units_}
        
        # POS_LABELS: for each unit, maps to which feature labels are positive
        # at least once when the unit fires (used to speed up beam search)
        pos_labels = {u: None for u in units_}
        g["pos_labels"] = pos_labels
        g["all_uidx"] = all_uidx
        g["all_uhitidx"] = all_uhitidx
        g["all_uhitidx_img"] = {u: None for u in units_}

        with mp.Pool(settings.PARALLEL) as p, tqdm(
            total=units, desc="Tallying units"
        ) as pbar:
            for (u, uidx, uhitidx, uhits, uhitidx_img, uhits_img) in p.imap_unordered(
                NeuronOperator.get_uhits, mp_args
            ):
                all_uidx[u] = uidx
                all_uhitidx[u] = uhitidx
                g["all_uhitidx_img"][u] = uhitidx_img
                # Get all labels which have at least one true here
                label_hits = mc.img2label[uidx].sum(0)
                # pos_labels[u] = np.argwhere(label_hits > 0).squeeze(1)
                pos_labs = np.argwhere(label_hits > 0).squeeze(1)
                pos_labels[u] = np.array([l for l in pos_labs if l in g["tally_labels"]])

                tally_units[u] = uhits
                g["tally_units_img"][u] = uhits_img
                pbar.update()

        # We don't need features anymore
        del g["features"]

        records = []

        if settings.UNIT_RANGE is None:
            ranger = range(units)
            nu = units
        else:
            ranger = settings.UNIT_RANGE
            nu = len(settings.UNIT_RANGE)
        
        wholeacts = features > threshold[np.newaxis, :, np.newaxis, np.newaxis]
        wholeacts = wholeacts.any((2, 3))

        mp_args = (
            (
                u, g["pos_labels"][u], g["masks"], g["all_uidx"][u], g["all_uhitidx"][u], 
                g["tally_units"][u], g["tally_units_img"][u], g["tally_labels"], 
                g["tally_labels_img"], g["labels"], data, g["data_mask_shape"]#, beam[u], curr_form_len
            )
            for u in ranger
        )
        
        with mp.Pool(settings.PARALLEL) as p, tqdm(
            total=nu, desc="IoU - primitives"
        ) as pbar:
            # for (u, best, best_noncomp, formulas) in p.imap_unordered(
            for (u, best_detacc_expl, netdissect, comp_expl) in p.imap_unordered(
                NeuronOperator.compute_best_iou, mp_args
            ):
                comp_expl_lab, comp_expl_iou, comp_expl_detacc = comp_expl
                netdissect_lab, netdissect_iou, netdissect_detacc = netdissect
                detacc_expl_lab, best_detacc_expl_iou, best_detacc_expl_detacc = best_detacc_expl

                comp_expl_name = comp_expl_lab.to_str(namer)
                comp_expl_cat = comp_expl_lab.to_str(cat_namer)
                netdissect_name = netdissect_lab.to_str(namer)
                netdissect_cat = netdissect_lab.to_str(cat_namer)
                detacc_expl_name = detacc_expl_lab.to_str(namer)
                detacc_expl_cat = detacc_expl_lab.to_str(cat_namer)

                r = {
                    "unit": u,
                    "category": detacc_expl_cat,
                    "label": detacc_expl_name,
                    "formula": detacc_expl_lab,
                    "detacc": round(best_detacc_expl_detacc, 4),
                    "iou": round(best_detacc_expl_iou, 4),
                    "length": len(detacc_expl_lab),

                    "comp_cat": comp_expl_cat,
                    "comp_label": comp_expl_name,
                    "comp_form": comp_expl_lab,
                    "comp_detacc": round(comp_expl_detacc, 4),
                    "comp_iou": round(comp_expl_iou, 4),

                    "netdissect_cat": netdissect_cat,
                    "netdissect_label": netdissect_name,
                    "netdissect_form": netdissect_lab,
                    "netdissect_detacc": round(netdissect_detacc, 4),
                    "netdissect_iou": round(netdissect_iou, 4),
                }

                records.append(r)

                pbar.update()
                
                if len(records) % 16 == 0:
                    tally_df = pd.DataFrame(records)
                    tally_df.to_csv(csvpath, index=False)
      
        tally_df = pd.DataFrame(records)
        tally_df.to_csv(csvpath, index=False)

        return records, mc

    @staticmethod
    def compute_best_iou(args):
        """
            Compute best concept and IoU for the given unit via beam search.

            :param args: tuple whose elements are the required parameters for the function

            :returns: (unit_number: int, (best_formula: F.F, best_iou: float),
                (best_noncomp_formula: F.F), (best_noncomp_iou: float))
                where noncomp indicates primitive formulas
        """
        (
            u, upos_labels, gmasks, uall_uidx, uall_uhitidx, 
            utally_units, utally_units_img, gtally_labels, 
            gtally_labels_img, glabels, data, data_mask_shape
        ) = args
        
        namer = lambda name: 'NONE' if name is None else data.name(None, name)
        
        ious = {}
        for lab in upos_labels:
            masks = gmasks[lab]
            lab_iou = NeuronOperator.compute_iou(
                uall_uidx, uall_uhitidx, masks,
                utally_units, gtally_labels[lab],
            )
            ious[lab] = lab_iou

        nonzero_iou = Counter({lab: iou for lab, iou in ious.items() if iou > 0})
        if not nonzero_iou:  # Nothing found
            return u, (F.Leaf(None), 0.0, 0.0, []), (F.Leaf(None), 0.0, 0.0), (F.Leaf(None), 0.0, 0.0), {}

        # Define candidates for beam search 
        if settings.BEAM_SEARCH_LIMIT is not None:
            # Restrict possible candidates 
            bs_labs = [
                t[0] for t in nonzero_iou.most_common(settings.BEAM_SEARCH_LIMIT)
            ]
        else:
            # Search with all possible labels 
            bs_labs = glabels
        
        formulas = {
            F.Leaf(lab): iou for lab, iou in nonzero_iou.most_common(settings.BEAM_SIZE)
        }
        # Best netdissect explanation
        netdissect = Counter(formulas).most_common(1)[0]
        
        detacc = NeuronOperator.compexpl_detectacc(
            netdissect[0], gmasks, uall_uhitidx, gtally_labels_img, data_mask_shape
        )
        best_detacc_expl = copy.deepcopy(netdissect)
        netdissect = (*netdissect, detacc)
        
        i, best_detacc, detacc_flag, comp_expl = 0, detacc, True, None

        # ==== BEAM SEARCH ====
        while i < settings.COMP_MAX_FORMULA_LENGTH - 1 or detacc_flag:
            new_formulas = {}
            for formula in formulas:
                for label in bs_labs:
                    for op, negate in [(F.Or, False), (F.And, False), (F.And, True)]:
                        new_term = F.Leaf(label)
                        if negate:
                            new_term = F.Not(new_term)
                        new_term = op(formula, new_term)
                        masks_comp = get_mask_global(gmasks, new_term)
                        comp_tally_label = cmask.area(masks_comp)
                        
                        comp_iou = NeuronOperator.compute_iou(
                            uall_uidx, uall_uhitidx, masks_comp, utally_units, comp_tally_label,
                        )

                        new_formulas[new_term] = comp_iou

            formulas.update(new_formulas)
            # Trim the beam
            formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))

            current_comp_expl = Counter(formulas).most_common(1)[0]
            current_detacc = NeuronOperator.compexpl_detectacc(
                current_comp_expl[0], gmasks, uall_uhitidx, gtally_labels_img, data_mask_shape
            )

            if i < settings.COMP_MAX_FORMULA_LENGTH - 1:
                comp_expl = copy.deepcopy(current_comp_expl)
                comp_detacc = copy.copy(current_detacc)
                i += 1
            
            # Permitting expl. with improved detacc or the same, as long as it's not just stancking up NOT
            if current_detacc > best_detacc or (current_detacc == best_detacc and "(NOT " not in current_comp_expl[0].to_str(namer)):
                best_detacc = current_detacc
                best_detacc_expl = copy.deepcopy(current_comp_expl)
            else:
                detacc_flag = False

        comp_expl = (*comp_expl, comp_detacc)
        best_detacc_expl = (*best_detacc_expl, best_detacc)

        return u, best_detacc_expl, netdissect, comp_expl

    @staticmethod
    def compute_iou(uidx, uhitidx, masks, tally_unit, tally_label):
        # Compute intersections
        tally_both = cmask.area(cmask.merge((masks, uhitidx), intersect=True))
        iou = (tally_both) / (tally_label + tally_unit - tally_both + 1e-10)

        return iou
    
    @staticmethod
    def compexpl_detectacc(formula, gmasks, uall_uhitidx, gtally_labels_img, data_mask_shape, no_singles=True):
        if isinstance(formula, F.Leaf):
            val = formula.val
            masks = gmasks[val]
            tally_lbl_img = gtally_labels_img[val]
        else:
            masks = get_mask_global(gmasks, formula)
            masks_np = cmask.decode(masks)
            masks_np = masks_np.reshape(data_mask_shape)
            tally_lbl_img = np.einsum('i->', np.any(masks_np, axis=(1,2)), dtype=int)

        if tally_lbl_img > 0.0:
            overlap = cmask.decode(cmask.merge((masks, uall_uhitidx), intersect=True))
            overlap = overlap.reshape(data_mask_shape)
            overlap = np.einsum('i->', np.any(overlap, axis=(1,2)), dtype=int)
            detacc = round(overlap/tally_lbl_img, 4)
        else:
            overlap, detacc = 0.0, 0.0

        if no_singles:
            return detacc
        
        if isinstance(formula, F.Leaf):
            single_accs = [detacc]
        else:
            single_accs = []
            for val in formula.get_vals():
                mask = gmasks[val]
                c_overlap = cmask.decode(cmask.merge((mask, uall_uhitidx), intersect=True))
                c_overlap = c_overlap.reshape(data_mask_shape)
                c_overlap = sum(np.any(c_overlap, axis=(1,2)))
                acc = c_overlap/gtally_labels_img[val]
                single_accs.append(round(acc, 4))
        
        return detacc, single_accs

    def tally(self, features, threshold, savepath="", full_savepath=None):
        if full_savepath is not None:
            csvpath = full_savepath
            savepath = full_savepath
        else:
            csvpath = os.path.join(settings.OUTPUT_FOLDER, savepath)

        return NeuronOperator._tally(
            features, self.data, threshold,
            0, self.data.size(),
            savepath, csvpath,
        )
