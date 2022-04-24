# %matplotlib inline 
'''
60-72% accuracy 4/23/22
from:
https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)
import joblib
from skimage.io import imread
from skimage.transform import resize


def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})women images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    if(im.shape == (80,80,3)):
                        data['label'].append(subdir[:])
                        data['filename'].append(file)
                        data['data'].append(im)
 
        joblib.dump(data, pklname)
data_path = fr'{os.getenv("HOME")}/../10sN1s/'
print(os.listdir(data_path))

base_name = 'flawedNflawless'
width = 80
 
include = {'10s', '1s'}
 
resize_all(src=data_path, pklname=base_name, width=width, include=include)

from collections import Counter
 
data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))

Counter(data['label'])

# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])
 
# set up the matplotlib figure and axes, based on the number of labels
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(8,4)
fig.tight_layout()
 
# make a plot for every label (equipment) type. The index method returns the 
# index of the first item corresponding to its search string, label in this case
for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
     
    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)

X = np.array(data['data'])
y = np.array(data['label'])
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#infor on the stratKfold:
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
# skf = StratifiedKFold(shuffle = False, n_splits= 3, random_state=None)
# stratKfold = skf.get_n_splits(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.1, 
    shuffle=True,
    random_state=42
    # stratify = stratKfold
)
# plt.show()
import math
def showTestImg(X_test, y_test):
    result_figsize_resolution = 40 # 1 = 100px

    images_list = X_test
    images_count = len(images_list)
    print('Images count: ', images_count)

    # Calculate the grid size:
    grid_size = math.ceil(math.sqrt(images_count))

    # Create plt plot:
    fig1, axes = plt.subplots(grid_size, grid_size, figsize=(result_figsize_resolution, result_figsize_resolution))

    current_file_number = 0
    for image in images_list:
        x_position = current_file_number % grid_size
        y_position = current_file_number // grid_size

        plt_image = images_list[current_file_number]
        axes[x_position, y_position].imshow(plt_image)
        
        print("position: ", x_position, y_position)
        print(y_test[current_file_number])
        print((current_file_number + 1), '/', images_count)
        current_file_number += 1
    plt.show()

def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
     
    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
     
    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'
         
    xtemp = np.arange(len(unique))
     
    plt.bar(xtemp + n*width, counts, align='center', alpha=.5, width=width)

    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)
 
plt.suptitle('relative amount of photos per type')
plot_bar(y_train, loc='left')
plot_bar(y_test, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train)), 
    'test ({0} photos)'.format(len(y_test))
])
#plt.show()

'''
convert colour images to grayscale, 
calculate their HOGs and finally scale the data.
three transformers in a row: RGB2GrayTransformer, 
HOGTransformer
StandardScaler. 
final result: array with a HOG for every image in the input
preprocess our RGB images to scaled HOG features
'''
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from skimage.io import imread

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

'''
pre process data
'''

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
 
# create an instance of each transformer
grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(8, 8), 
    cells_per_block=(4, 4), 
    orientations=9, 
    block_norm='L2-Hys'
)
scalify = StandardScaler()
 
# call fit_transform on each transform converting X_train step by step
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)
#y is unused and set to 576 for some reason (164, 576)
print(X_train_prepared.shape)

'''
The next step: train a classifier
Stochastic Gradient Descent (SGD),fast and works reasonably well.
'''
	
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_prepared, y_train)

'''
testthe trained SGD classifier,use test set. 
First, we transform it using the same transformers as before. 
ensure test data will not influence the transformers. 
Some transformers, like PCA (Principle Component Analysis), can optimise themselves on the data before applying the transformation. 
To prevent this, call transform and not fit_transform on the test data.
'''
X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)
'''make a prediction for test set and look at the results. 
How many of the predictions match with y_test?'''
y_pred = sgd_clf.predict(X_test_prepared)
# print(np.array(y_pred == y_test)[:25])
print('')
print("prediction length: ", len(y_pred))
print("test length: ", len(y_test))
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

import pandas as pd
# put the data in a dataframe for pretty printing
df = pd.DataFrame(
    np.c_[y_test, y_pred], 
    columns=['true_label', 'prediction']
)
print(df)

'''
confusion matrix
'''

from sklearn.metrics import confusion_matrix
# predictions = y_pred
# lables = y_test
label_names = ['10s', '1s']
cmx = confusion_matrix(y_test, y_pred, labels=label_names)
df = pd.DataFrame(cmx, columns=label_names, index=label_names)
df.columns.name = 'prediction'
df.index.name = 'label'
# print(df)
# plt.show()
plt.imshow(cmx)
plt.xticks([])
plt.yticks([])
plt.colorbar()
#plt.show()
cmx = confusion_matrix(y_test, y_pred)
print(cmx)

'''optimization '''
from sklearn.pipeline import Pipeline
from sklearn import svm
 
HOG_pipeline = Pipeline([
    ('grayify', RGB2GrayTransformer()),
    ('hogify', HogTransformer(
        pixels_per_cell=(12, 14), 
        cells_per_block=(5, 4), 
        orientations=9, 
        block_norm='L2-Hys')
    ),
    ('scalify', StandardScaler()),
    ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
])
 
clf = HOG_pipeline.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Percentage correct: ', 100*np.sum(clf.predict(X_test) == y_test)/len(y_test))
df = pd.DataFrame(
    np.c_[y_test, y_pred], 
    columns=['true_label', 'prediction']
)
print(df)
plt.imshow(cmx)
plt.xticks([])
plt.yticks([])
plt.colorbar()
#plt.show()
cmx = confusion_matrix(y_test, y_pred)
print(cmx)
from sklearn.model_selection import GridSearchCV
 
param_grid = [
    {
        'hogify__orientations': [8, 9],
        'hogify__cells_per_block': [(3, 3), (5, 4)],
        'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 14)]
    },
    {
        'hogify__orientations': [8],
         'hogify__cells_per_block': [(5, 4)],
         'hogify__pixels_per_cell': [(12, 14)],
         'classify': [
             SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
             svm.SVC(kernel='linear')
         ]
    }
]

grid_search = GridSearchCV(HOG_pipeline, 
                           param_grid, 
                           cv=3,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=1,
                           return_train_score=True)
 
grid_res = grid_search.fit(X_train, y_train)


# save the model to disk
joblib.dump(grid_res, 'hog_sgd_model.pkl')


# description of the best performing object, a pipeline in our case.
print(grid_res.best_estimator_)

# the highscore during the search
print("best score: ", grid_res.best_score_)

pp.pprint(grid_res.best_params_)

best_pred = grid_res.predict(X_test)
print('Percentage correct: ', 100*np.sum(best_pred == y_test)/len(y_test))
showTestImg(
    X_test=X_test , y_test=y_test
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
 
def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]
         
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
     
plot_confusion_matrix(cmx)
 
# the types appear in this order
print('\n', sorted(np.unique(y_test)))
cmx_svm = confusion_matrix(y_test, best_pred)
plot_confusion_matrix(cmx, vmax1=225, vmax2=100, vmax3=12)
plt.show()