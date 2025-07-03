#!/usr/bin/env python
from __future__ import annotations

"""main logical entrypoint for ReX."""

import copy
import os
import sys
import time
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch as tt
from PIL import Image
from scipy.io import loadmat
from sqlalchemy.orm import Session
from tqdm import trange  # type: ignore

from rex_xai.explanation.evaluation import Evaluation
from rex_xai.explanation.explanation import Explanation
from rex_xai.explanation.multi_explanation import MultiExplanation
from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.input.onnx import get_prediction_function
from rex_xai.output.database import update_database
from rex_xai.responsibility.prediction import Prediction, default_prediction_function
from rex_xai.responsibility.resp_maps import ResponsibilityMaps
from rex_xai.responsibility.responsibility import causal_explanation
from rex_xai.utils._utils import ReXDataError, ReXScriptError, Strategy
from rex_xai.utils.logger import logger


def try_preprocess(args: CausalArgs, model_shape: Tuple[int], device: tt.device):
    """Makes an attempt to preprocess input data as required for the model.

    Data preprocessing is based on file extension and (possibly) user-defined mode.
    File extensions in ``[".jpg", ".jpeg", ".png", ".tif", ".tiff"]`` are treated
    as images, ".npy" and ".mat" are treated as Numpy arrays, and ".nii" are treated as nifti files.
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

        img = Image.open(args.path)
        if img.mode == "RGBA":
            logger.warning("RGBA input image provided, converting to RGB")
            img = img.convert("RGB")

        data = Data(img, model_shape, device)
        data.generic_image_preprocess(means=args.means, stds=args.stds, norm=args.norm)

    # a numpy "npy" array or matlab "mat" file
    elif ext in (".npy", ".mat"):
        if args.mode in ("tabular", "spectral"):
            if ext == ".mat":
                raw_data = np.load(loadmat(args.path)["val"])
            else:
                raw_data = np.load(args.path)
            data = Data(raw_data, model_shape, mode=args.mode, device=device)
            data.data = data.generic_tab_preprocess()
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

    Uses a custom preprocesssing function if this is defined in ``args.script.preprocess``,
    otherwise :py:func:`~rex_xai.explanation.try_preprocess()`.

    Args:
        model_shape: shape of the input tensor of the model, as returned by
            :py:func:`~rex_xai.explanation.get_prediction_func_from_args()`
        device: as returned by :py:func:`~rex_xai._utils.get_device()`
        args: configuration values for ReX

    Returns:
        Data: the processed input data

    """
    if args.script is not None:
        if hasattr(args.script, "preprocess"):
            data = args.script.preprocess(args.path, model_shape, device)
            if args.context_location is not None:
                data.context = args.script.preprocess(
                    args.context_location, model_shape, device
                ).data.to(device)
        else:
            raise ReXScriptError(
                f"{args.script_location} is missing a preprocess() function"
            )
    else:
        # no custom preprocessing, so we make our best guess as to what to do
        data = try_preprocess(args, model_shape, device)
        if args.context_location is not None:
            logger.warning(
                f"{args.context_location} is not gonna be used, since ReX doesn't know how to process"
            )
            args.context = False
            args.mask_value = 0  # Setting it to a default value in this case
    return data


def predict_target(
    data: Data, args: CausalArgs, prediction_func
) -> Prediction | list[Prediction]:
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
        targets_str = ''.join(f"{t.classification}\n" for t in target)
        logger.info(f"Found {len(target)} targets, the targets found are: \n{targets_str}")
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
    data: Data,
    args: CausalArgs,
    prediction_func,
    # keep_all_maps=False,
    custom_height=None,
    custom_width=None,
) -> tuple[ResponsibilityMaps, dict]:
    """Calculates ResponsibilityMaps for input data using given args.

    Runs :py:func:`~rex_xai.responsibility.causal_explanation` for ``args.iters`` iterations,
    and returns a ResponsibilityMaps object and a dictionary containing some statistics about
    the calculation process.
    The ResponsibilityMaps object by default only includes the responsibility map that matches
    the classification of the input data. Set ``keep_all_maps`` to ``True`` to retain all maps.

    Args:
        data: processed input data object
        args: configuration values for ReX
        prediction_func: prediction function for the model
        keep_all_maps: whether to retain all :py:class:`~rex_xai.resp_maps.ResponsibilityMaps`,
            or just the one that matches the target classification.

    Returns:
        tuple containing

        - ResponsibilityMaps: ResponsibilityMaps for the given data, prediction function, and args.
        - dict: statistics for the call of this function that generated the ResponsibilityMaps object
    """

    if isinstance(data.target, list):
        if any(t.classification is None for t in data.target):
            raise ValueError(
                "No target classification found in the list of targets. Please run `predict_target` before running `calculate_responsibility`."
            )
    else:
        if data.target is None or data.target.classification is None:
            raise ValueError(
                "No target classification found. Please run `predict_target` before running `calculate_responsibility`."
            )

    maps = ResponsibilityMaps(style=args.responsibility_style)
    if custom_height is not None and custom_width is not None:
        maps.new_map(data.target.classification, custom_height, custom_width)
    elif data.model_height is not None:
        maps.new_map(
            data.target.classification,
            data.model_height,
            data.model_width,
            data.model_depth,
        )
    else:
        maps.new_map(data.target.classification, data.model_height, data.model_width)

    total_passing: int = 0
    total_failing: int = 0
    max_depth_reached: int = 0
    avg_box_size: float = 0.0

    if args.iters > 0:
        for i in trange(args.iters, disable=not args.progress_bar):
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

    return maps, run_stats


def analyze(exp: Explanation, data_mode: str | None) -> Dict[str, float]:
    """Analyzes an Explanation.

    Analyzes the area ratio, entropy difference, insertion and deletion curves for an
    Explanation object, prints them, and returns them.

    Args:
        exp: Explanation object as returned by :py:func:`~rex_xai.explanation._explanation`
        data_mode: Mode of the input data. Entropy of the responsibility map calculated if ``data_mode``
            is "RGB". If ``data_mode'' is ``spectral'' then spectral entropy is calculated.

    Returns:
        tuple containing

        - area (float)
        - entropy (float)
        - insertion_curve (float)
        - deletion_curve (float)

    """
    eval = Evaluation(exp)

    rat = eval.ratio()

    good, bad = eval.robustness()
    ent = None
    max_ent = None
    if data_mode == "RGB":
        ent = eval.responsibility_entropy()  # type: ignore
    elif data_mode == "spectral":
        ent, max_ent = eval.spectral_entropy()

    iauc, dauc = eval.insertion_deletion_curve(
        exp.prediction_func, normalise=exp.args.normalise_curves
    )

    analysis_results = {
        "area": rat,
        "entropy": ent,
        "robustness": good / (good + bad),
        "max_entropy": max_ent,
        "insertion_curve": iauc,
        "deletion_curve": dauc,
    }

    return analysis_results


def _explanation(
    args: CausalArgs,
    model_shape: Tuple[int],
    prediction_func: Callable,
    device: tt.device,
    db: Session | None = None,
    path: str | None = None,
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
    data.set_mask_value(args.mask_value)

    logger.debug(
        "args.mask_value is %s, data.mask_value is %s", args.mask_value, data.mask_value
    )

    data.target = predict_target(data, args, prediction_func)

    time_taken = 0
    start = time.time()

    logger.info("Calculating responsibility map")
    resp_object, run_stats = calculate_responsibility(data, args, prediction_func)
    if args.negative_responsibility:
        resp_object.negative_responsibility(data.target.classification)
    mid = time.time()
    logger.info(f"Finished building responsibility map after {mid - start} seconds")

    logger.info("Extracting explanation from responsibility map")
    clauses = None
    exp = None
    if args.strategy == Strategy.MultiSpotlight:
        exp = MultiExplanation(resp_object, prediction_func, data, args, run_stats)
        if not args.no_extract:
            exp.extract()

            clauses = exp.separate_by(args.permitted_overlap)
            logger.info(f"found the following sets of explanations {clauses}")

            logger.info(f"keeping only {clauses[0]}")
            clauses = clauses[0]
    else:
        exp = Explanation(resp_object, prediction_func, data, args, run_stats)
        if not args.no_extract:
            exp.extract()

    assert exp is not None
    results = None
    if args.analyse is not None:
        if args.strategy == Strategy.MultiSpotlight:
            logger.warning("still to write")
            pass
        else:
            logger.info("Analysing explanation")
            results = analyze(exp, data.mode)
            end = time.time()
            time_taken = end - start

            if data.mode == "spectral":
                print(
                    f"INFO:ReX:classification {exp.data.target.classification}, area {results['area']}, responsibility entropy {results['entropy']},",  # type: ignore
                    f"max entropy {results['max_entropy']}",
                )
            else:
                if args.analyse == "print":
                    print(
                        f"INFO:ReX:path {args.path}, classification {exp.data.target.classification}, area {results['area']}, responsibility entropy {results['entropy']}, robustness {results['robustness']}",  # type: ignore
                        f"insertion curve {results['insertion_curve']}, deletion curve {results['deletion_curve']}, time {time_taken}",
                    )
                else:
                    assert exp.data.target is not None
                    with open(args.analyse, "a") as out:
                        out.write(
                            f"{args.path},{exp.data.target.classification},{results['area']},{results['entropy']},{results['robustness']},{results['insertion_curve']},{results['deletion_curve']},{time_taken}\n"
                        )

    else:
        end = time.time()
        time_taken = end - start
        logger.info(f"Time taken: {time_taken:.2f}s")

    if args.surface is not None:
        if path is not None:
            pass
        elif args.surface == "show":
            path = None
        else:
            path = args.surface
        logger.info(f"Surface plot is saved at {path}")
        exp.surface_plot(path)

    if args.heatmap is not None:
        if args.heatmap == "show":
            path = None
        else:
            path = args.heatmap
        logger.info(f"Heatmap plot is saved at {path}")
        exp.heatmap_plot(path)

    if args.output is not None:
        if path is None:
            if args.output == "show":
                path = None
            else:
                path = args.output
        if args.strategy == Strategy.MultiSpotlight:
            exp.save(path, clauses=clauses)  # type: ignore
        else:
            exp.save(path)  # type: ignore

    if db is not None:
        if args.strategy == Strategy.MultiSpotlight:
            logger.info("writing multiple explanations to database")
            update_database(db, exp, time_taken, multi=True, clauses=clauses)
        else:
            logger.info("writing to database")
            update_database(db, exp, time_taken, analysis_results=results)

    if data.device == "mps":
        with tt.no_grad():
            tt.mps.empty_cache()
    elif data.device == "cuda":
        with tt.no_grad():
            tt.cuda.empty_cache()

    return exp


def get_prediction_func_from_args(args: CausalArgs):
    """Takes a CausalArgs object and gets the prediction function and model shape.

    If ``args.script`` specifies a prediction function and model shape, returns these.
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
    prediction_func = None
    model_shape = None
    if hasattr(args.script, "prediction_function"):
        prediction_func = args.script.prediction_function  # type: ignore
    if hasattr(args.script, "model_shape"):
        model_shape = args.script.model_shape  # type: ignore
    else:
        ps = get_prediction_function(args)
        if ps is None:
            raise RuntimeError("Unable to create an onnx inference instance")
        else:
            prediction_func, model_shape = ps

    if prediction_func is None:
        if hasattr(args.script, "model"):
            prediction_func = default_prediction_function(args.script.model)  # type: ignore
        else:
            raise ReXDataError("ReX cannot find a valid prediction function")
    if model_shape is None:
        raise ReXDataError("ReX cannot find a valid model shape")
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

    prediction_func, model_shape = get_prediction_func_from_args(args)

    if isinstance(model_shape[0], int) and model_shape[0] < args.batch_size:
        logger.warning(
            f"Resetting batch size to size of model's first axis: {model_shape[0]}"
        )
        args.batch_size = model_shape[0]

    # directory of data to process
    if os.path.isdir(args.path):
        explanations: List[Explanation] = []
        dir = args.path
        path = None
        for dir, _, files in os.walk(args.path):
            for f in files:
                to_process = os.path.join(dir, f)
                logger.info("processing %s", to_process)
                # TODO can we remove this copy?
                current_args = copy.copy(args)
                current_args.path = to_process
                if args.output is not None and args.output != "show":
                    out_dir = os.path.dirname(args.output)
                    name, ext = os.path.splitext(args.output)
                    fname, _ = os.path.splitext(f)
                    path = f"{out_dir}_{fname}_{name}{ext}"
                exp = _explanation(
                    current_args,
                    model_shape,
                    prediction_func,
                    device,
                    db,
                    path=path,
                )
                if exp is not None:
                    explanations.append(exp)
        return explanations

    else:
        # a single explanation
        return _explanation(args, model_shape, prediction_func, device, db)
