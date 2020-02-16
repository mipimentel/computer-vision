import cv2 as cv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog


class DeSkewTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, size=28):
        self.size = size
        self.affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

    def de_skew(self, img):
        m = cv.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * self.size * skew], [0, 1, 0]])
        img = cv.warpAffine(img, M, (self.size, self.size), flags=self.affine_flags)
        return img

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([self.de_skew(img) for img in X])


class HogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bin_n=8, ppc=(7, 7), cells_per_block=(4, 4)):
        self.bin_n = bin_n
        self.ppc = ppc
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([hog(img, orientations=self.bin_n, pixels_per_cell=self.ppc,
                             cells_per_block=self.cells_per_block, block_norm='L2', visualize=False) for img in X])

if __name__== '__main__':
    import tensorflow as tf
    from sklearn import svm

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    hog_pipeline = Pipeline([
        ('deskew', DeSkewTransformer()),
        ('HOG', HogTransformer()),
    ])

    mnist_transform_train = hog_pipeline.fit_transform(x_train)
    mnist_transform_test = hog_pipeline.fit_transform(x_test)
    clf = svm.SVC(kernel='rbf', gamma=1, C=10).fit(mnist_transform_train, y_train)


