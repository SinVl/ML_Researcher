# ML_Researcher

ML_Researcher is a Python module in which you can learn internal machine learning and understand the mathematical foundations that are used in ML

## Installation

### Dependencies

- Python (>= 3.5)
- NumPy (>= 1.11.0)

## Usage

```python
import numpy as np
from linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
lr = LinearRegression().fit(X, y)
reg.predict(np.array([[3, 5]]))
array([16.])
```