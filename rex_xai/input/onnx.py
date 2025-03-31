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
from rex_xai.responsibility.prediction import Prediction, from_pytorch_tensor
from rex_xai.input.input_data import Setup

from rex_xai.utils.logger import logger


class OnnxRunner:
    def __init__(self, session: InferenceSession, setup: Setup, device) -> None:
        self.session = session
        self.input_shape = session.get_inputs()[0].shape
        self.output_shape = session.get_outputs()[0].shape
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.setup: Setup = setup
        self.device: str = device

    def run_on_cpu(
        self,
        tensors: Union[tt.Tensor, List[tt.Tensor]],
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
            tensors = np.stack([t.detach().cpu().numpy() for t in tensors])  # type: ignore

        preds = []

        try:
            prediction = self.session.run(None, {self.input_name: tensors})[0]
            for i in range(0, prediction.shape[0]):
                confidences = softmax(prediction[i])
                if raw:
                    for i in range(len(self.output_shape) - len(confidences.shape)):
                        confidences = np.expand_dims(confidences, axis=0)
                    return confidences
                if binary_threshold is not None:
                    if confidences[0] >= binary_threshold:
                        classification = 1
                    else:
                        classification = 0
                    tc = confidences[0]
                else:
                    classification = np.argmax(confidences)
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
        self,
        tensors,
        device,
        tsize,
        binary_threshold,
        raw=False,
        device_id=0,
        target=None,
    ):
        # input_shape = self.session.get_inputs()[0].shape # Gets the shape of the input (e.g [batch_size, 3, 224, 224])
        batch_size = len(tensors) if isinstance(tensors, list) else tensors.shape[0]

        if isinstance(tensors, list):
            tensors = [m.contiguous() for m in tensors]
            shape = tuple(
                [batch_size] + list(self.input_shape)[1:]
            )  # batch_size + remaining input shape
            ptr = tensors[0].data_ptr()

        else:
            tensors = tensors.contiguous()
            shape = tuple(tensors.shape)
            ptr = tensors.data_ptr()

        binding = self.session.io_binding()
        binding.bind_input(
            name=self.input_name,
            device_type=device,
            device_id=device_id,
            element_type=np.float32,
            shape=shape,
            buffer_ptr=ptr,
        )

        output_shape = [batch_size] + list(self.output_shape[1:])

        z_tensor = tt.empty(output_shape, dtype=tt.float32, device=device).contiguous()

        binding.bind_output(
            name=self.output_name,
            device_type=device,
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(z_tensor.shape),
            buffer_ptr=z_tensor.data_ptr(),
        )

        self.session.run_with_iobinding(binding)
        if raw:
            return z_tensor
        return from_pytorch_tensor(z_tensor, target=target)

    def gen_prediction_function(self):
        if self.device == "cpu" or self.setup == Setup.ONNXMPS:
            return (
                lambda tensor,
                target=None,
                raw=False,
                binary_threshold=None: self.run_on_cpu(
                    tensor, target, raw, binary_threshold
                ),
                self.input_shape,
            )
        if self.device == "cuda":
            return (
                lambda tensor,
                target=None,
                device=self.device,
                raw=False,
                binary_threshold=None: self.run_with_data_on_device(
                    tensor,
                    device,
                    len(tensor),
                    binary_threshold,
                    raw=raw,
                    target=target,
                ),
                self.input_shape,
            )


def get_prediction_function(args):
    # def get_prediction_function(model_path, gpu: bool, logger_level=3):
    sess_options = ort.SessionOptions()

    ort.set_default_logger_severity(args.ort_logger)

    # are we (trying to) run on the gpu?
    if args.gpu:
        logger.info("using gpu for onnx inference session")
        if platform.uname().system == "Darwin":
            # note this is only true for M+ chips
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            sess_options.intra_op_num_threads = args.intra_op_num_threads
            sess_options.inter_op_num_threads = args.inter_op_num_threads
            device = "mps"
            _, ext = os.path.splitext(os.path.basename(args.model))
            # for the moment, onnx does not seem to support data copying on mps, so we fall back to
            # copying data to the cpu for inference
            setup = Setup.ONNXMPS
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            device = "cuda"
            setup = Setup.PYTORCH

        # set up sesson with gpu providers
        sess = ort.InferenceSession(
            args.model, sess_options=sess_options, providers=providers
        )  # type: ignore
    # are we running on cpu?
    else:
        logger.info("using cpu for onnx inference session")
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(
            args.model, sess_options=sess_options, providers=providers
        )
        device = "cpu"
        setup = Setup.PYTORCH

    onnx_session = OnnxRunner(sess, setup, device)

    return onnx_session.gen_prediction_function()
