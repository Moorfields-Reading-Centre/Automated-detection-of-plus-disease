import os
import csv
import json
from PIL import Image
import numpy as np

from keras.applications import DenseNet201
from keras.models import Model
from keras.layers import MaxPool2D, Dense, Conv2D, Flatten, Softmax
from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score
import batch_preparation

data_folder = # folder with images

# 1st column filename, 2nd column fold, last column grading
grading_file = # filename of csv file with grading

model_folder = # folder to save models to

# names of classes in grading csv, 1-hot encoding
grading_labels = {
    'normal': [1, 0, 0],
    'pre-plus': [0, 1, 0],
    'plus': [0, 0, 1],
}

input_size = 512, 512
batch_size = 6
learning_rate = 1e-5
n_iterations_per_epoch = 100
n_epochs = 100
n_classes = len(grading_labels)

fold_names = 'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'test', 'excluded'

with open(grading_file, 'r') as f:
    split = list(csv.reader(f))
    header = split[0]
    data = split[1:]

# fold_files structure: {fold_name: [(filename, grading), ...]}
# using group label (row[-1])
fold_files = {
    fold_name: [(row[0], row[-1]) for row in data if row[1] == fold_name]
    for fold_name in fold_names
}

# load all images into memory (takes a while)
all_files = [d[0] for d in data]
all_data = {
    path: np.array(Image.open(os.path.join(data_folder, path)))[:, :, :3]
    for path in all_files
}


def get_grading_data(folds, grading_labels):
    
    # all files and gradings in folds
    files = [
        (path, grading)
        for fold_name, fold_items in fold_files.items()
        for path, grading in fold_items 
        if fold_name in folds
    ]
    
    # map grading to list of raw images (np array)
    result = {
        grading: [all_data[path] for path, g in files if grading == g]
        for grading in grading_labels
    }
    
    return files, result


def get_data(validation_fold, grading_labels):
    training_folds = {
        f for f in fold_names if f not in (validation_fold, 'test', 'excluded')
    }
    
    training_files, training_data = get_grading_data(training_folds, grading_labels)
    validation_files, validation_data = get_grading_data((validation_fold, ), grading_labels)
    
    # no overlap in files between training and validation
    assert not (set(training_files) & set(validation_files))
    
    def get_patient(filename):
        return tuple(filename.split('-')[:-1])
    
    training_patients = {get_patient(filename) for filename, grading in training_files}
    validation_patients = {get_patient(filename) for filename, grading in validation_files}
    
    # no overlap in patients between training and validation
    assert not (set(training_patients) & set(validation_patients))

    return training_data, validation_data


def get_model():
    # use pre-trained model
    base_model = DenseNet201(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(*input_size, n_classes),
        pooling=None,
    )
    # add layers at the end of the model
    l = MaxPool2D()(base_model.output)
    l = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(l)
    l = Flatten()(l)
    l = Dense(units=n_classes)(l)
    l = Softmax()(l)
    model = Model(base_model.input, l)
    
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model


def avg_auc(scores, grading_labels):
    return np.mean([scores[k] for k in grading_labels])


def validate(model, grading_labels, data_set_preprocessed):
    
    n = len(data_set_preprocessed)
    yt = np.zeros((n, 3))
    yp = np.zeros((n, 3))
    
    for i, (grading, x) in enumerate(data_set_preprocessed):
        yt[i] = grading_labels[grading]
        yp[i] = model.predict_on_batch(x[np.newaxis])[0]
           
    scores = {}
    for name, labels in grading_labels.items():
        i = np.argmax(labels)
        try:
            scores[name] = roc_auc_score(yt[:,i], yp[:,i])
        except:
            scores[name] = np.nan
    
    scores['avg_auc'] = avg_auc(scores, grading_labels)
    
    correct = np.argmax(yt, axis=-1) == np.argmax(yp, axis=-1)
    scores['accuracy'] = np.count_nonzero(correct) / n

    return scores


def get_batches(validation_fold, grading_labels):
    training_data, validation_data = get_data(validation_fold, grading_labels)

    training_batch_preparation = batch_preparation.BatchPreparationClassBalanced(
        grading_labels, 
        training_data, 
        n_channels=3,
        batch_size=batch_size, 
    )
    
    # validation samples will be reused for each validation step
    validation_set_preprocessed = [
        (grading, batch_preparation.sample_patch(image, augment=False))
        for grading, images in validation_data.items() 
        for image in images
    ]

    return training_batch_preparation, validation_set_preprocessed


def training_loop(validation_fold, grading_labels):
    
    model = get_model()
    
    training_batch_preparation, validation_set_preprocessed = get_batches(
        validation_fold, grading_labels
    )
    
    losses = []
    all_scores = {
        k: [] for k in grading_labels
    }
    all_scores['accuracy'] = []
    all_scores['avg_auc'] = []

    best_normal_auc = 0
    best_avg_auc = 0
    
    for epoch in range(n_epochs):
        for i in range(n_iterations_per_epoch):
            x, y = training_batch_preparation.get_batch()
            loss = model.train_on_batch(x, y)
            losses.append(loss)
           
        scores = validate(model, grading_labels, validation_set_preprocessed)
        
        improvement = False
        if scores['normal'] > best_normal_auc:
            best_normal_auc = scores['normal']
            improvement = True
        if scores['avg_auc'] > best_avg_auc:
            best_avg_auc = scores['avg_auc']
            improvement = True
        
        # save model if validation score improved (after warm-up period)
        if improvement and epoch > 15:
            model.save(
                '{}/val_fold_{}_epoch_{}'.format(model_folder, validation_fold, epoch)
            )
        
        for k, v in scores.items():
            all_scores[k].append(v)

    return model, losses, all_scores     


for f in range(5):
    validation_fold = 'fold_{}'.format(f)
    export_file = 'scores_{}.json'.format(validation_fold)
        
    print('starting training loop', validation_fold)
    model, losses, all_scores = training_loop(validation_fold, grading_labels)
    with open(export_file, 'w') as f:
        json.dump({
            'losses': [float(l) for l in losses],
            'all_scores': all_scores
        }, f)