## Script usage
ReX can take in scripts that define the model behaviour, the preprocessing for the model and how the model's output can be interpreted by ReX. This is to allow the users to provide custom preprocessing/models to ReX.

As outlined in the [command line section](command_line.md), the user can pass in the script using the `--script` argument.

```bash
rex imgs/dog.jpg --script scripts/pytorch.py -vv --output dog_exp.jpg 
```

### Contents of the python script
There are three main components to the script:
- A preprocess function -> preprocess(path, shape, device, mode) -> Data
- A function that calls the model -> prediction_function(mutants, target=None, raw=False, binary_threshold=None):
- Model shape function -> model_shape() -> Tuple  

#### Preprocessing function

The preprocessing function is responsible for loading the image and transforming it to the correct shape for the model.

```python
def preprocess(path, shape, device, mode) -> Data:
```
The **key steps** in the preprocess function are:
- Load the data from the path
- Transform the data to requirements of the model
- Return a Data object


The function should return a Data object. The Data object contains the following fields:
- `input` -> The raw input
- `data` -> The transformed input for the model
- `model_shape` -> The shape of the model input
- `device` -> The device the data is on e.g. "cuda"
- `mode` -> The mode of the data e.g. "RGB", "L", "voxel"
- `process` -> A boolean that indicates whether the data mode should be accessed or not
- `model_height` -> The height of the model input
- `model_width` -> The width of the model input
- `model_height` -> The height of the model input 
- `model_channels` -> The number of channels in the model input
- `transposed` -> Whether the data is transposed or not
- `model_order` -> The order of the model input e.g. "first" or "last"
- `background` -> The value of the background of the image e.g. 0 or 255 ... etc. For a range of values, use a tuple e.g. (0, 255)
- `context` -> The context of the image e.g. the specific background like a beach or a road that can be used as an occlusion if specified as mask value

The Data object can be initialised with the `input`, `model_shape`, `device` and optionally the `mode` and `process`.:
Example:
```python
data = Data(input, model_shape, device, mode="voxel", process=False)
# Set the other attributes of the Data object separately like so
data.model_height = 224
```


#### Prediction function

The prediction function is responsible for running inference on the model, processing and returning the output.

```python
def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
```

**Parameters**:
- `mutants` -> A list of mutants to run inference on?
- `target` -> The target class 
- `raw` -> Whether to return the raw output (e.g. the probability of the classification) or not 
- `binary_threshold` -> The threshold for binary classification e.g. 0.5

**Returns**:
- A list of Prediction objects or a float if raw is True

The Prediction object contains the following fields:
 - `classification` -> The classification of the mutant: Optional[int]
 - `confidence` -> The confidence of the classification: Optional[float]
 - `bounding_box` -> The bounding box for the classification: Optional[NDArray]
 - `target` -> The target class: Optional[int]
 - `target_confidence` -> The confidence of the target class: Optional[float]

#### Model shape function

The model shape function is responsible for returning the shape of the model input.

```python
def model_shape() -> []:
```
**Example:**
```python
def model_shape():
    return ["N", 3, 224, 224]
```

---
Example scripts can be found in the `tests/scripts` and `scripts` directory.