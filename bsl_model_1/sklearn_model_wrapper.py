import numpy as np

class SklearnKerasWrapper:
    def __init__(self, model, classes):
        self.model = model
        self.classes = classes

    def predict(self, input_data, verbose=0):
        if input_data.ndim > 2:
            input_data = input_data.reshape((input_data.shape[0], -1))

        try:
            probs = self.model.predict_proba(input_data)
        except AttributeError:
            preds = self.model.predict(input_data)
            probs = np.zeros((len(preds), len(self.classes)))
            for i, pred in enumerate(preds):
                probs[i, pred] = 1.0
        return probs
