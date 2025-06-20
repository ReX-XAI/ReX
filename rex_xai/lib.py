from PIL import Image

from rex_xai.explanation.explanation import Explanation
from rex_xai.explanation.rex import calculate_responsibility, predict_target
from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.output import visualisation
from rex_xai.responsibility.prediction import default_prediction_function


class ReX:
    def __init__(
        self,
        model,
        model_shape,
        device,
        mode,
        prediction_function=None,
    ) -> None:
        self.args = CausalArgs()
        self.model = model
        self.model_shape = model_shape
        self.device = device
        self.mode = mode
        self.explanation = None
        self.prediction_function = prediction_function
        self.data = None
        self.maps = None
        self.stats = None

        if self.prediction_function is None:
            self.get_default_prediction_function()

    def set_tabular_data(self, path, data=None):
        if self.mode == "spectral":
            self.args.path = path
            self.args.mode = "spectral"
            if data is not None:
                self.data = Data(
                    data,
                    data.shape,
                    self.device,
                    mode=self.args.mode,
                    process=True,
                )

    def set_rgb_image(self, path):
        if self.mode == "RGB":
            self.args.path = path
            self.args.mode = "RGB"
            img = Image.open(self.args.path).convert("RGB")
            self.data = Data(img, self.model_shape, self.device, mode="RGB")
            return img

    def show_target(self):
        if self.data is not None and self.data.target is not None:
            print(self.data.target)
        else:
            print("a target has not yet been set")

    def set_transformed_data(self, transformed_data):
        if self.data is not None:
            self.data.data = transformed_data

    def get_default_prediction_function(self):
        self.prediction_function = default_prediction_function(self.model)

    def set_target(self):
        if self.data is not None:
            self.data.target = predict_target(self.data, self.prediction_function)

    def set_prediction_function(self, function):
        self.prediction_function = function

    def calculate_responsibility(self, args=None):
        if args is None:
            args = self.args
        if self.data is not None:
            if self.data.mask_value is None:
                self.data.set_mask_value(args.mask_value)
            maps, stats = calculate_responsibility(
                self.data, args, self.prediction_function
            )
            self.maps = maps
            self.stats = stats
            return maps, stats

    def get_explanation(self, maps=None, stats=None):
        if maps is None:
            maps = self.maps
        if stats is None:
            stats = self.stats
        if self.data is not None:
            self.explanation = Explanation(
                maps, self.prediction_function, self.data, self.args, stats
            )

    def extract(self):
        if self.explanation is not None:
            self.explanation.extract()
        else:
            self.get_explanation

    def rerun_with(self, new_args: CausalArgs):
        self.calculate_responsibility(args=new_args)

    def show(self):
        if self.mode == "spectral":
            visualisation.spectral_plot(
                self.explanation.sufficiency_mask,
                self.data,
                self.maps.get(self.data.target.classification),
                self.args.heatmap_colours,
            )
