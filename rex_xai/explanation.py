#!/usr/bin/env python3
# from __future__ import annotations
"""main logical entrypoint for ReX."""

import copy
import os
import sys
import time
from typing import List, Tuple, Union

import numpy as np
import torch as tt
from PIL import Image
from sqlalchemy.orm import Session
from tqdm import trange  # type: ignore

from rex_xai.config import CausalArgs
from rex_xai.database import update_database
from rex_xai.evaluation import Evaluation
from rex_xai.extraction import Explanation
from rex_xai.input_data import Data
from rex_xai.logger import logger
from rex_xai.onnx import get_prediction_function
from rex_xai.resp_maps import ResponsibilityMaps
from rex_xai.responsibility import causal_explanation
from rex_xai.prediction import Prediction


def try_preprocess(args: CausalArgs, model_shape: Tuple[int], device: tt.device):
    """Makes an attempt to preprocess input data as required for the model.

    Data preprocessing is based on file extension and (possibly) user-defined mode.
    File extensions in ``[".jpg", ".jpeg", ".png", ".tif", ".tiff"]`` are treated
    as images, ".npy" are treated as Numpy arrays, and ".nii" are treated as nifti files.
    For any other file extension, we create a ``Data`` object without pre-processing.

    Args:
        args: configuration values for ReX
        model_shape: shape of the input tensor of the model, as returned by
            :py:func:`~rex_xai.explanation.get_prediction_func_from_args()`
        device: as returned by :py:func:`~rex_xai._utils.get_device()`

    Returns:
      Data: the processed input data
    """
    _, ext = os.path.splitext(args.path)
    # args.path is an image
    if ext.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        # a simple sanity check: spectral_occlusion is not suitable for images
        # so remind the user to update rex.toml and set mask_value to 0 in the meantime
        try:
            if args.mask_value in ("tabular", "spectral"):  # type: ignore
                logger.warning(
                    f"{args.mask_value} is not suitable for images. Changing mask_value to 0"
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


def load_and_preprocess_data(
    model_shape: Tuple[int], device: tt.device, args: CausalArgs
):
    """Loads input data from filepath and does preprocessing.

    Uses a custom preprocesssing function if this is defined in ``args.custom.preprocess``,
    otherwise :py:func:`~rex_xai.explanation.try_preprocess()`.

    Args:
        model_shape: shape of the input tensor of the model, as returned by
            :py:func:`~rex_xai.explanation.get_prediction_func_from_args()`
        device: as returned by :py:func:`~rex_xai._utils.get_device()`
        args: configuration values for ReX

    Returns:
        Data: the processed input data

    """
    if args.custom is not None and hasattr(args.custom, "preprocess"):
        data = args.custom.preprocess(args.path, model_shape, device, mode=args.mode)
    else:
        # no custom preprocessing, so we make our best guess as to what to do
        data = try_preprocess(args, model_shape, device)

    return data


def predict_target(data: Data, prediction_func) -> Prediction:
    """Predicts classification of input data, using given prediction function.

    Uses ``prediction_func`` to identify the classification of the input data and return
    this as the target classification for ReX.

    Args:
        data: processed input data object
        args: configuration values for ReX
        prediction_func: prediction function for the model

    Returns:
        Prediction: the predicted target classification and confidence
    """
    target = prediction_func(data.data, None)

    if isinstance(target, list):
        target = target[0]

    if target is not None:
        logger.info(
            "image classified as %s with %f confidence",
            target.classification,
            target.confidence,
        )
    else:
        logger.warning("no target found")
        sys.exit(-1)

    return target


def calculate_responsibility(
    data: Data, args: CausalArgs, prediction_func, keep_all_maps=False
) -> Explanation:
    """Calculates an Explanation for input data using given args.

    Runs :py:func:`~rex_xai.responsibility.causal_explanation` for ``args.iters`` iterations,
    and returns an Explanation object.
    The resulting Explanation object by default only includes the responsibility map that matches
    the classification of the input data. Set ``keep_all_maps`` to ``True`` to retain all maps.
    The  Explanation object also includes some statistics about the calculation process, in the
    ``run_stats`` field.

    Args:
        data: processed input data object
        args: configuration values for ReX
        prediction_func: prediction function for the model
        keep_all_maps: whether to retain all :py:class:`~rex_xai.resp_maps.ResponsibilityMaps`,
            or just the one that matches the target classification.

    Returns:
        Explanation: Explanation for the given data, prediction function, and args.
    """

    if data.target is None or data.target.classification is None:
        raise ValueError(
            "No target classification found. Please run `predict_target` before running `calculate_responsibility`."
        )

    maps = ResponsibilityMaps()
    maps.new_map(data.target.classification, data.model_height, data.model_width)

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
                current_map=maps.get(data.target.classification),
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

    run_stats = {
        "total_passing": total_passing,
        "total_failing": total_failing,
        "max_depth_reached": max_depth_reached,
        "avg_box_size": avg_box_size,
    }

    exp = Explanation(
        maps, prediction_func, data, args, run_stats, keep_all_maps=keep_all_maps
    )

    return exp


def analyze(exp: Explanation, data_mode: str | None):
    """Analyzes an Explanation.

    Analyzes the area ratio, entropy difference, insertion and deletion curves for an
    Explanation object, prints them, and returns them.

    Args:
        exp: Explanation object as returned by :py:func:`~rex_xai.explanation._explanation`
        data_mode: Mode of the input data. Entropy difference is only calculated if ``data_mode``
            is one of ["RGB", "L"].

    Returns:
        tuple containing

        - area (float)
        - entropy_diff (float)
        - insertion_curve (float)
        - deletion_curve (float)

    """
    eval = Evaluation(exp)
    rat = eval.ratio()
    if data_mode in ("RGB", "L"):
        be, ae = eval.entropy_loss()  # type: ignore
        ent = be - ae
    elif data_mode in ("spectral", "tabular"):
        ent = eval.spectral_entropy()
    else:
        ent = None

    iauc, dauc = eval.insertion_deletion_curve(exp.prediction_func)

    analysis_results = {
        "area": rat,
        "entropy_diff": ent,
        "insertion_curve": iauc,
        "deletion_curve": dauc,
    }

    return analysis_results


def _explanation(
    args: CausalArgs,
    model_shape: Tuple[int],
    prediction_func,
    device: tt.device,
    db: Session | None = None,
):
    """Takes a CausalArgs object and model information and returns a Explanation.

    Takes a CausalArgs object, model shape and prediction function and returns an Explanation.
    Depending on the input ``args``, optionally produces output plots, analyses the output
    explanation, and/or writes results to a database.

    Args:
        args: configuration values for ReX
        model_shape: shape of the input tensor of the model, as returned by :py:func:`~rex_xai.explanation.get_prediction_func_from_args()`
        prediction_func: as returned by :py:func:`~rex_xai.explanation.get_prediction_func_from_args()`
        device: as returned by :py:func:`~rex_xai._utils.get_device()`
        db: None or as returned by :py:func:`~rex_xai.database.initialise_rex_db()`

    Returns:
        Explanation:
            An :py:class:`~rex_xai.extraction.Explanation` object containing the causal reponsibility explanation
            calculated using the given ``args``.

    """
    data = load_and_preprocess_data(model_shape, device, args)
    data.set_mask_value(args.mask_value, device=data.device)
    logger.debug("args.mask_value is %s, data.mask_value is %s", args.mask_value, data.mask_value)

    data.target = predict_target(data, prediction_func)

    start = time.time()

    exp = calculate_responsibility(data, args, prediction_func)
    exp.extract(args.strategy)

    if args.analyze:
        results = analyze(exp, data.mode)
        print(
            f"INFO:ReX:area {results['area']}, entropy {results['entropy_diff']},",
            f"insertion curve {results['insertion_curve']}, deletion curve {results['deletion_curve']}",
        )

    end = time.time()
    time_taken = end - start
    logger.info(time_taken)

    if args.surface is not None:
        if args.surface == "show":
            path = None
        else:
            path = args.surface
        exp.surface_plot(path)

    if args.heatmap is not None:
        if args.heatmap == "show":
            path = None
        else:
            path = args.heatmap
        exp.heatmap_plot(path)

    if args.output is not None:
        if args.output == "show":
            path = None
        else:
            path = args.output
        exp.save(path)

    if db is not None:
        logger.info("writing to database")
        update_database(
            db,
            data.target,  # type: ignore
            exp,
            time_taken,
            exp.run_stats["total_passing"],
            exp.run_stats["total_failing"],
            exp.run_stats["max_depth_reached"],
            exp.run_stats["avg_box_size"],
        )

    return exp


def validate_args(args: CausalArgs):
    """Validates a CausalArgs object.

    Checks that ``args.path`` is not None.

    Args:
        args: configuration values for ReX
    """

    if args.path is None:
        raise FileNotFoundError("Input file path cannot be None")


def get_prediction_func_from_args(args: CausalArgs):
    """Takes a CausalArgs object and gets the prediction function and model shape.

    If ``args.custom`` specifies a prediction function and model shape, returns these.
    Otherwise gets the prediction function and model shape from the provided model
    file.

    Args:
        args: configuration values for ReX

    Returns:
        tuple containing

        - ``prediction_func``
        - ``model_shape``

    Raises:
        RuntimeError: if an onnx inference instance cannot be created from the provided model file.

    """
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


def explanation(
    args: CausalArgs, device: tt.device, db: Session | None = None
) -> Union[Explanation, List[Explanation]]:
    """Takes a CausalArgs object and returns a Explanation.

    Takes a CausalArgs object and returns either an Explanation, or a list of Explanations
    if the input ``args.path`` is a directory rather than a path to a single file.

    Args:
        args: configuration values for ReX
        device: as returned by :py:func:`~rex_xai._utils.get_device()`
        db: None or as returned by :py:func:`~rex_xai.database.initialise_rex_db()`

    Returns:
        Explanation:
            An :py:class:`~rex_xai.extraction.Explanation` object containing the causal reponsibility explanation
            calculated using the given ``args``.

    """

    validate_args(args)

    prediction_func, model_shape = get_prediction_func_from_args(args)

    if isinstance(model_shape[0], int) and model_shape[0] < args.batch:
        logger.warning(
            f"Resetting batch size to size of model's first axis: {model_shape[0]}"
        )
        args.batch = model_shape[0]

    # multiple explanations
    if os.path.isdir(args.path):
        dir = args.path
        explanations = []
        for dir, _, files in os.walk(args.path):
            for f in files:
                to_process = os.path.join(dir, f)
                logger.info("processing %s", to_process)
                current_args = copy.copy(args)
                current_args.path = to_process
                explanations.append(
                    _explanation(current_args, model_shape, prediction_func, device, db)
                )
        return explanations

    else:
        # a single explanation
        return _explanation(args, model_shape, prediction_func, device, db)
