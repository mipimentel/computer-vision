from sklearn.pipeline import Pipeline

# Models to be tested
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.metrics import classification_report
from ml_utils.transformers import DeSkewTransformer, HogTransformer

import ml_utils as mlu

# distributions for random search
from scipy.stats import randint, expon, reciprocal
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="grid_search",
    help="Models to test, if specified grid search will make a search space with all models",
)
ap.add_argument(
    "-s", "--scoring", type=str, default="accuracy", help="Scoring metric to be used"
)
args = vars(ap.parse_args())

# TODO: make grid to other classifiers like, GaussianNB and Logistic Regression
randomized_params = {
    "KNeighborsClassifier": {"n_neighbors": randint(low=1, high=30)},
    "RandomForest": {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    },
    "SVM": {
        "kernel": ["linear", "rbf"],
        "C": reciprocal(0.1, 200000),
        "gamma": expon(scale=1.0),
    },
}

models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
}


if __name__ == "__main__":
    import tensorflow as tf

    model_name = args["model"]
    scoring = args["scoring"]

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    hog_pipeline = Pipeline(
        [("deskew", DeSkewTransformer()), ("HOG", HogTransformer())]
    )

    mnist_transform_train = hog_pipeline.fit_transform(x_train, y_train)
    mnist_transform_test = hog_pipeline.fit_transform(x_test, y_test)

    # for faster grid search only 5000 samples will be used for training
    X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(
        mnist_transform_train, y_train, test_size=0.91666, random_state=42
    )

    print(
        "Doing randomized grid search on {model} with scoring: {scoring}".format(
            model=model_name, scoring=scoring
        )
    )
    grid = RandomizedSearchCV(
        models[model_name],
        param_distributions=randomized_params[model_name],
        n_iter=100,
        scoring=scoring,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    grid.fit(X_train_min, y_train_min)

    scores = cross_val_score(
        grid.best_estimator_,
        mnist_transform_train,
        y_train,
        cv=5,
        scoring=scoring,
        verbose=0,
        n_jobs=-1,
    )

    CV_scores = scores.mean()
    STDev = scores.std()
    Test_scores = grid.score(mnist_transform_test, y_test)

    metadata = {
        "Model_Name": model_name,
        "Parameters": grid.best_params_,
        "Test_Score": Test_scores,
        "CV_Mean": CV_scores,
        "CV_STDEV": STDev,
        "CV_Scores": scores,
    }

    clf = grid.best_estimator_.fit(mnist_transform_train, y_train)
    clf.score(mnist_transform_test, y_test)
    y_pred = clf.predict(mnist_transform_test)
    clf_report = classification_report(y_test, y_pred)
    mlu.save(clf, **metadata, clf_report=clf_report, name=model_name)
