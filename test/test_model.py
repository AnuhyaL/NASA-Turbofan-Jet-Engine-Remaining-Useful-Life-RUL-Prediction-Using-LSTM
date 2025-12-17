import numpy as np
from models import RULPredictor

def test_model_output_shape():
    X = np.random.randn(5, 50, 10)
    y = np.random.randn(5)

    model = RULPredictor(input_size=10)
    model.fit(X, y, epochs=1)
    preds = model.predict(X)

    assert len(preds) == len(y)
