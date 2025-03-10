# PyLobe
A NumPy-native library for wrangling with far-field antenna patterns.

Functionality for:
- Rotation
- Interpolation
- Normalization
- Slicing/masking
- Padding
- Beamwidth calculation
- Visualization

## Installation
### From VCS url
```
uv pip install "git+https://github.com/saeedece/pylobe"
```

### From local source
```
git clone https://github.com/saeedece/pylobe.git && cd pylobe
uv pip install .
```

## Usage
PyLobe is organized into seven sub-modules (`pylobe.beamwidth`, `pylobe.interpolate`, `pylobe.pattern`, `pylobe.transform`, `pylobe.utils`, `pylobe.visualize`, `pylobe.transform`) and has no top-level exports. Functions should be imported from sub-modules as needed.

Example:

```python
from pylobe.beamwidth import compute as compute_beamwidth
from pylobe.transform import rotate_pattern_bilinear
from pylobe.utils import Axis
...
```


## To-do
- Support for batched interpolation.
- More analytical expressions in `pylobe.pattern`.
