import joblib  # for saving models from skikit-learn


def save(model, cv_info, classification_report, name="model"):
    _model = {"cv_info": cv_info, "classification_report": classification_report, "model": model}
    joblib.dump(_model, "models_saved/" + name + ".pkl")


def load(name="model", verbose=True, with_metadata=False):
    _model = joblib.load("models_saved/" + name + ".pkl")
    if verbose:
        print("\nLoading model with the following info:\n")
        [print("{key}: {val}".format(key=key, val=val)) for key, val in _model["cv_info"].items()]
        print("\nClassification Report:\n")
        print(_model["classification_report"])
    if not with_metadata:
        return _model["model"]
    else:
        return _model
