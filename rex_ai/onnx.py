#!/usr/bin/env python

"""onnx model management"""

from typing import Optional, Union, List
import sys
import os
import torch as tt
import platform
from scipy.special import softmax
import numpy as np

import onnxruntime as ort
from onnxruntime import InferenceSession
from rex_ai.prediction import Prediction, from_pytorch_tensor
from rex_ai.input_data import Setup

from rex_ai.logger import logger


def run_on_cpu(
    session: InferenceSession,
    tensors: Union[tt.Tensor, List[tt.Tensor]],
    input_name: str,
    target: Optional[Prediction],
    raw: bool,
    binary_threshold: Optional[float] = None,
):
    """Convert a pytorch tensor, or list of tensors, to numpy arrays on the cpu for onnx inference."""
    # check if it's a single tensor or a list of tensors
    if isinstance(tensors, list):
        tensor_size = tensors[0].shape[0]
    else:
        tensor_size = tensors.shape[0]

    if tensor_size == 1:
        tensors = tensors.detach().cpu().numpy()  # type: ignore
    else:
        tensors = np.stack(
            [t.squeeze(0).detach().cpu().numpy() for t in tensors]
        )  # type: ignore

    preds = []

    try:
        prediction = session.run(None, {input_name: tensors})[0]
        for i in range(0, prediction.shape[0]):
            confidences = softmax(prediction[i])
            if raw:
                return confidences[0]
            if binary_threshold is not None:
                if confidences[0] >= binary_threshold:
                    classification = 1
                else:
                    classification = 0
                tc = confidences[0]
            else:
                classification = np.argmax(confidences)
                # print(prediction[i], confidences, classification)
                if target is not None:
                    tc = confidences[target.classification]
                else:
                    tc = None
            preds.append(
                Prediction(
                    classification,
                    confidences[classification],
                    None,
                    target=target,
                    target_confidence=tc,
                )
            )

        return preds
    except Exception as e:
        logger.fatal(e)
        sys.exit(-1)


def run_with_data_on_device(
    session, tensors, input_name, device, tsize, binary_threshold
):
    if isinstance(tensors, list):
        # TODO this should probably be a stack
        tensors = [m.contiguous() for m in tensors]
        shape = tuple(
            (
                len(tensors),
                3,
                224,
                224,
            )
        )
        ptr = tensors[0].data_ptr()
    else:
        tensors = tensors.contiguous()
        shape = tuple(tensors.shape)
        ptr = tensors.data_ptr()

    binding = session.io_binding()
    binding.bind_input(
        name=input_name,
        device_type=device,
        device_id=0,
        element_type=np.float32,
        shape=shape,
        buffer_ptr=ptr,
    )

    z_tensor = tt.empty(
        [tsize, 1000], dtype=tt.float32, device=device
    ).contiguous()

    binding.bind_output(
        name=session.get_outputs()[0].name,
        device_type=device,
        device_id=0,
        element_type=np.float32,
        shape=tuple(z_tensor.shape),
        buffer_ptr=z_tensor.data_ptr(),
    )

    session.run_with_iobinding(binding)
    return from_pytorch_tensor(z_tensor)


def get_prediction_function(model_path, gpu: bool):
    sess_options = ort.SessionOptions()
    # are we (trying to) run on the gpu?
    if gpu:
        logger.info("using gpu for onnx inference session")
        if platform.uname().system == "Darwin":
            providers = ["CoreMLExecutionProvider"]
            device = "mps"
            _, ext = os.path.splitext(os.path.basename(model_path))
            # for the moment, onnx does not seem to support data copying on mps, so we fall back to
            # copying data to the cpu for inference
            setup = Setup.ONNXMPS
        else:
            providers = ["CUDAExecutionProvider"]
            device = "cuda"
            setup = Setup.PYTORCH

        # set up sesson with gpu providers
        sess = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )  # type: ignore
    # are we running on cpu?
    else:
        logger.info("using cpu for onnx inference session")
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        device = "cpu"
        setup = Setup.PYTORCH

    input_name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    logger.info("model shape %s", shape)

    if device == "cpu" or setup == Setup.ONNXMPS:
        return (
            lambda x, target=None, raw=False, binary_threshold=None: run_on_cpu(
                sess, x, input_name, target, raw, binary_threshold
            ),
            shape,
        )
    if device == "cuda":
        return (
            lambda x,
            target=None,
            device=device,
            binary_threshold=None: run_with_data_on_device(
                sess, x, input_name, device, len(x), binary_threshold
            ),
            shape,
        )
