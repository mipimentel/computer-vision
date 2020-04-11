import os
import json
import joblib  # for saving models from skikit-learn
import numpy as np

dirname = os.path.dirname(__file__)
models_path = os.path.abspath(os.path.join(dirname, "../models_saved"))


class MLUtilsError(BaseException):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# transform numpy arrays to list for json encoding
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def create_folder(path):
    if os.path.exists(path) and os.path.isdir(path):
        raise MLUtilsError(
            f"model with the name '{os.path.basename(path)}' already exists."
        )
    else:
        os.mkdir(path)
        return path


def save(model, **kwargs):
    r"""Saves the model with its metadata.


        This method creates a directory with the model and its metada saved as pkl and json format.

        Parameters
        ----------
        model : Scikit-Learn Estimator
            estimator from Scikit-Learn
        Keyword Arguments : Keyword Arguments
            the metadata to be saved, python objects and numpy array supported for decoding in json metadata
            if "name" not specified in Keyword Arguments, it will be saved as "model" by default.

        Returns
        -------
        out : None
            returns None.

        Raises
        ------
        MLUtilsError
            If model name already exists in saving.

        See Also
        --------
        ml_utils.utils.load

        Examples
        --------
        In this example we alread have a trained classifier, defined as 'clf', and will calculate some metrics we
        would want to save as metadata.

        >>> import ml_utils as mlu
        >>> # calculates cross validation score and declares it as cv_score
        >>> # calculates classification report and declares it as clf_report
        >>> mlu.save(clf, cv_score=cv_score, clf_report=clf_report, name=model_name)
        """

    # the dumps followed by loads is for Encoding correct numpy arrays while
    # keeping the json format in the file instead of a single string
    _metadata = json.loads(json.dumps(kwargs, cls=NumpyArrayEncoder, sort_keys=True))
    name = kwargs.get("name", "model")
    _model_path = create_folder(os.path.join(models_path, name))
    with open(os.path.join(_model_path, "metadata.json"), "w") as json_file:
        json.dump(_metadata, json_file)
    joblib.dump(model, os.path.join(_model_path, f"{name}.pkl"))


def load(name="model", verbose=True, with_metadata=True):
    r"""loads the model with its metadata.

        This method loads a the model and its metada saved as pkl and json format.

        Parameters
        ----------
        name : basestring
            model name to load
        verbose : bool
            prints the model metadata if True.
        with_metadata : bool
            returns the metadata dict if True, if False returns None.

        Returns
        -------
        model : first output
        metadata : second output

        Raises
        ------
        MLUtilsError
            If model name does not exist.

        See Also
        --------
        ml_utils.utils.save

        Examples
        --------
        In this example we have a saved trained classifier, saved with the name 'classifier', and we will load it.

        >>> import ml_utils as mlu
        >>> model_name = 'classifier'
        >>> model, model_metadata = mlu.load(model_name, verbose=True, with_metadata=True)
        """
    # name = kwargs.get("name", "model")
    _model_path = os.path.join(models_path, name)
    if os.path.exists(_model_path) and os.path.isdir(_model_path):
        _model = joblib.load(os.path.join(_model_path, f"{name}.pkl"))
        if verbose:
            with open(os.path.join(_model_path, "metadata.json"), "r") as read_file:
                metadata = json.load(read_file)
            print("\nLoading model with the following info:\n")
            [print(f"{key}: {val}") for key, val in metadata.items()]
        if with_metadata:
            return _model, metadata
        else:
            return _model, None
    else:
        raise MLUtilsError(
            f"model with the name '{os.path.basename(_model_path)}' does not exists."
        )
