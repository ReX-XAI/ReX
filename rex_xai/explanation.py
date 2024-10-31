#!/usr/bin/env python3
# from __future__ import annotations
"""main logical entrypoint for ReX."""

import os
import sys
import time
from tqdm import trange  # type: ignore
from typing import List, Tuple, Union
import torch as tt
import numpy as np
from PIL import Image

from rex_xai.evaluation import Evaluation
from rex_xai.extraction import Explanation
from rex_xai.responsibility import causal_explanation
from rex_xai.input_data import Data
from rex_xai.onnx import get_prediction_function
from rex_xai.resp_maps import ResponsibilityMaps
from rex_xai.occlusions import set_mask_value
from rex_xai.config import CausalArgs
from rex_xai.logger import logger, set_log_level
from rex_xai.database import initialise_rex_db, update_database


def try_preprocess(args: CausalArgs, model_shape: Tuple[int], device: str):
    """Makes an attempt to preprocess data based on file extension and (possibly)
    user defined mode.

    @param args: CausalArgs object
    @param model_shape: shape of the input tensor of the model

    @return a <Data> object
    """
    _, ext = os.path.splitext(args.path)
    # args.path is an image
    if ext.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        # a simple sanity check: spectral_occlusion is not suitable for images
        # so remind the user to update rex.toml and set mask_value to 0 in the meantime
        try:
            if args.mask_value in ("tabular", "spectral"):  # type: ignore
                logger.warning(
                    "tabular is not suitable for images. Changing mask_value to 0"
                )
                args.mask_value = 0
        except AttributeError:
            pass

        # if not args.processed:
        data = Data(Image.open(args.path), model_shape, device)
        # data = Data(Image.open(args.path).convert("RGB"), model_shape, device)
        data.generic_image_preprocess(means=args.means, stds=args.stds, norm=args.norm)

    # a compressed numpy array file
    elif ext in ".npy":
        if args.mode in ("tabular", "spectral"):
            data = Data(np.load(args.path), model_shape, mode=args.mode, device=device)
            data.data = tt.from_numpy(data.generic_tab_preprocess()).to(device)
        else:
            logger.fatal("we do not generically handle this datatype")
            return NotImplemented
    # nifti files for 3d data
    elif ext in ".nii":
        logger.fatal("we do not (yet) handle nifti files generically")
        return NotImplemented
    else:
        # we don't know what to do!
        data = Data(args.path, model_shape, mode=args.mode, device=device)

    return data

def load_and_preprocess_data(model_shape, device, args):

    if hasattr(args.custom, "preprocess"):
        data = args.custom.preprocess(args.path, model_shape, device, mode=args.mode)
    else:
        # no custom preprocessing, so we make our best guess as to what to do
        data = try_preprocess(args, model_shape, device)

    return data

def predict_target(data, args, prediction_func):
    target = prediction_func(data.data, None)

    if isinstance(target, list):
        target = target[0]
    if target is not None:
        data.classification = target.classification

    if args.target is None:
        args.target = target

    if args.target is not None:
        logger.info(
            "image classified as %s with %f confidence",
            args.target.classification,
            args.target.confidence,
        )
    else:
        logger.warning("no target found")
        sys.exit(-1)

    return target

def calculate_responsibility(data, args, prediction_func):

    maps = ResponsibilityMaps()
    maps.new_map(args.target.classification, data.model_height, data.model_width)

    total_passing: int = 0
    total_failing: int = 0
    max_depth_reached: int = 0
    avg_box_size: float = 0.0

    if args.iters > 0:
        for i in trange(args.iters, disable=not args.progress):
            (
                local_maps,
                passing,
                failing,
                depth_reached,
                avg_bs,  # average box size
            ) = causal_explanation(
                i + 1,
                data,
                args,
                prediction_func,
                current_map=maps.get(args.target.classification),
            )

            total_passing += passing
            total_failing += failing
            max_depth_reached = max(max_depth_reached, depth_reached)
            avg_box_size += avg_bs
            # TODO this needs to be smarter. If we only have shallow penetration, then
            # this is doing us a disservice. Perhaps leave merging until after completion
            # of all iterations. Might potentially use a lot of memeory though
            if depth_reached > 1:
                maps.merge(local_maps)

    avg_box_size /= args.iters

    logger.info(
        "Total Passing: %d, Total Failing %d, Max Depth Reached %d, Avg Box Size %f",
        total_passing,
        total_failing,
        max_depth_reached,
        avg_box_size,
    )

    return maps, total_passing, total_failing, max_depth_reached, avg_box_size

def analyze(data, exp, prediction_func, args):
    eval = Evaluation(exp)
    rat = eval.ratio()
    if data.mode in ("RGB", "L"):
        be, ae = eval.entropy_loss()  # type: ignore
        ent = be - ae
    else:
        ent = None

    iauc, dauc = eval.insertion_deletion_curve(prediction_func)
    if args.verbosity < 2:
        set_log_level(2, logger)
    logger.info(
        "area %f, entropy difference %f, insertion curve %f, deletion curve %f",
        rat,
        ent,
        iauc,
        dauc,
    )
    set_log_level(args.verbosity, logger)


def _explanation(args, model_shape, prediction_func, device, db=None):

    data = load_and_preprocess_data(model_shape, device, args)

    target = predict_target(data, args, prediction_func)

    args.mask_value = set_mask_value(args.mask_value, data, device=data.device)

    start = time.time()

    maps, total_passing, total_failing, max_depth_reached, avg_box_size = calculate_responsibility(data, args, prediction_func)

    exp = Explanation(maps, prediction_func, target, data, args)  # type: ignore

    exp.extract(args.strategy)

    if args.analyze:
        analyze(data, exp, prediction_func, args)

    end = time.time()
    time_taken = end - start
    logger.info(time_taken)

    if args.surface is not None:
        exp.surface_plot(maps)

    if args.heatmap is not None:
        exp.heatmap_plot(maps)

    if args.output is not None:
        exp.save()

    if db is not None:
        logger.info("writing to database")
        update_database(
            db,
            target,  # type: ignore
            exp,
            time_taken,
            total_passing,
            total_failing,
            max_depth_reached,
            avg_box_size,
        )

    return exp


def validate_args(args):
    if args.path is None:
        raise FileNotFoundError("Input file path cannot be None")


def get_prediction_func_from_args(args):
    if hasattr(args.custom, "prediction_function") and hasattr(
        args.custom, "model_shape"
    ):
        prediction_func = args.custom.prediction_function  # type: ignore
        model_shape = args.custom.model_shape()  # type: ignore
    else:
        ps = get_prediction_function(args.model, args.gpu)
        if ps is None:
            raise RuntimeError("Unable to create an onnx inference instance")
        else:
            prediction_func, model_shape = ps
    
    return prediction_func, model_shape


def explanation(args: CausalArgs, device, db) -> Union[Explanation, List[Explanation]]:
    """Take a CausalArgs object and return a Explanation.

    @param args: CausalArgs
    @return Explanation
    """

    validate_args(args)

    prediction_func, model_shape = get_prediction_func_from_args(args)

    if isinstance(model_shape[0], int) and model_shape[0] < args.batch:
        logger.warning(f"Resetting batch size to size of model's first axis: {model_shape[0]}")
        args.batch = model_shape[0]

    # multiple explanations
    if os.path.isdir(args.path):
        dir = args.path
        explanations = []
        for dir, _, files in os.walk(args.path):
            for f in files:
                to_process = os.path.join(dir, f)
                logger.info("processing %s", to_process)
                args.path = to_process
                explanations.append(
                    _explanation(args, model_shape, prediction_func, device, db)
                )
        return explanations

    else:
        # a single explanation
        return _explanation(args, model_shape, prediction_func, device, db)
