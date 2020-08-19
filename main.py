import random
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, confusion_matrix, \
    precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import constant
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tensorflow.keras.applications.densenet import DenseNet121
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import csv
from datetime import datetime
import pickle
import os
from glob import glob
from itertools import chain
from sklearn.utils import shuffle

import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import use, is_interactive

use('TkAgg')
# plt.ion()
print('Using', plt.get_backend(), 'as graphics backend.')
print('Is interactive:', is_interactive())

dataset_root = '/media/fanta/52A80B61A80B42C9/Users/fanta/datasets'


def load_dataset(file_name):
    # Read the dataset metadata
    metadata = pd.read_csv(file_name)
    metadata = metadata.drop(columns=metadata.columns[-1])

    # Add a 'path' column to the metadata dataframe holding the full path to dataset images
    all_image_paths = {os.path.basename(x): x for x in
                       glob(os.path.join(dataset_root + '/data', 'images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', metadata.shape[0])
    metadata['path'] = metadata['Image Index'].map(all_image_paths.get)

    # One-hot encode the findings for each row in the metadata dataframe, adding more columns as needed
    # TODO can this done in one line, with pandas? Check TF tutorials
    all_labels = np.unique(list(chain(*metadata['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    for c_label in all_labels:
        metadata[c_label] = metadata['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)

    metadata['class'] = metadata.Pneumonia.map(lambda value: 1 if value == 1 else 0)

    to_be_dropped = [col for col in metadata.columns if col not in ('Patient ID', 'path', 'class')]
    metadata.drop(columns=to_be_dropped, inplace=True)

    return metadata


def split_dataset(dataset, test_size):
    # Want to split the dataset in two partitions, stratified in the 'pneumonia' variable; but also keeping all samples
    # related to the same patient in the same partition.
    patient_id = 'Patient ID'
    pneumonia = 'class'
    # For each patient, count how many positive samples in the dataset, and shuffle the resulting serie
    patient_positive_count = shuffle(dataset.groupby(patient_id)[pneumonia].agg('sum'))
    # patient_positive_count.reset_index(inplace=True, drop=True)
    n_positive_samples = sum(dataset[pneumonia])
    n_wanted = int(n_positive_samples * test_size)
    selected_patients = set()
    n_selected = 0
    for id, count in patient_positive_count.items():
        selected_patients.add(id)
        n_selected += count
        if n_selected >= n_wanted:
            break
    # TODO Now I have enough positive cases, should I go on selecting patients with negative cases only until I als
    # have enough negative cases?
    test_set = shuffle(dataset[dataset[patient_id].isin(selected_patients)])
    test_set.reset_index(inplace=True, drop=True)
    training_set = shuffle(dataset[~dataset[patient_id].isin(selected_patients)])
    training_set.reset_index(inplace=True, drop=True)
    assert len(training_set) + len(test_set) == len(dataset)
    return training_set, test_set


def augment_positive_samples(file_names_df, output_file_name, params):
    file_names_df = file_names_df[file_names_df['class'] == 1]
    # print('Loaded information for', len(file_names_df), 'files with positive samples.')
    print('Making', params['augm_factor'], 'new samples for every positive sample.')

    for file in Path(params['augm_target_dir']).glob('*.png'):
        file.unlink()

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                                      shear_range=.2,
                                                                      zoom_range=(.75, 1.20),
                                                                      brightness_range=(.8, 1.2))

    # TODO try other interpolations, e.g. bicubic
    generated_data = image_generator.flow_from_dataframe(dataframe=file_names_df,
                                                         # directory=params['dataset_root'],
                                                         save_to_dir=params['augm_target_dir'],
                                                         target_size=params['image_size'],
                                                         x_col='path',
                                                         y_col='class',
                                                         class_mode='raw',
                                                         # Note: raw is needed for model.fit() to compute precision
                                                         batch_size=params['augm_batch_size'],
                                                         shuffle=False)
    wanted_batches = int(params['augm_factor'] * np.ceil(generated_data.samples / generated_data.batch_size))
    generated_images_count = 0
    for image_batch, label_batch in generated_data:
        assert sum(label_batch == 1) == len(image_batch)
        generated_images_count += len(image_batch)
        print('.', end='', flush=True)
        if generated_data.total_batches_seen == wanted_batches:
            break
    print('\nGenerated', generated_images_count, 'images in', params['augm_target_dir'])

    new_rows = []
    with open(output_file_name, 'w') as target_file:
        target_file.write('file_name,label,class\n')
        for file_name in Path(params['augm_target_dir']).glob('*.png'):
            to_be_written = params['augm_target_dir'] + '/' + str(file_name.name)
            target_file.write(to_be_written + ',51\n')
            new_rows.append({'Patient ID': -1, 'path': to_be_written, 'class': 1})
    print('Written metadata file', params['aug_metadata_file_name'])
    metadata_df = pd.DataFrame(new_rows)
    return metadata_df


def run_experiment(params):
    batch_size = int(params['batch_size'])
    augm_factor = params['augm_factor']
    test_set_fraction = params['test_set_fraction']
    n_epochs = params['n_epochs']
    val_batch_size = params['val_batch_size']
    augm_batch_size = params['augm_batch_size']
    theta = params['theta']
    image_size = params['image_size']
    image_shape = params['image_shape']
    dataset_root = params['dataset_root']
    checkpoints_dir = params['checkpoints_dir']
    checkpoints_path = params['checkpoints_path']
    augm_target_subdir = params['augm_target_subdir']
    augm_target_dir = params['augm_target_dir']
    aug_metadata_file_name = params['aug_metadata_file_name']
    val_results_file_name = params['val_results_file_name']
    py_seed = params['py_seed']
    np_seed = params['np_seed']
    tf_seed = params['tf_seed']

    np.random.seed(np_seed)
    tf.random.set_seed(tf_seed)
    random.seed(py_seed)

    samples_fig_no = 2

    file_names_df = load_dataset(dataset_root + '/data/Data_Entry_2017.csv')

    training_df, test_df = split_dataset(file_names_df, test_set_fraction)
    training_df, validation_df = split_dataset(training_df, test_set_fraction / (1 - test_set_fraction))
    if augm_factor != 0:
        file_names_augmented_df = augment_positive_samples(file_names_df=training_df,
                                                           output_file_name=aug_metadata_file_name,
                                                           params=params)
        training_df = pd.concat([training_df, file_names_augmented_df], axis='rows')
        training_df = training_df.sample(frac=1, replace=False)
        training_df.reset_index(inplace=True, drop=True)

    # training_df = training_df.astype({'class': 'int'})
    # validation_df = validation_df.astype({'class': 'int'})
    # test_df = test_df.astype({'class': 'int'})


if __name__ == '__main__':
    params = {'n_epochs': 3,  # TODO use a named tuple instead?
              'batch_size': hp.quniform('batch_size', 16, 32, 1),
              'val_batch_size': 64,
              'test_set_fraction': hp.uniform('test_set_fraction', .20, .30),
              'augm_batch_size': 64,
              'augm_factor': hp.choice('augm_factor', (1,)),
              'theta': .5,
              'image_shape': (224, 224, 3),
              'dataset_root': '/media/fanta/52A80B61A80B42C9/Users/fanta/datasets',
              'checkpoints_dir': 'checkpoints',
              'augm_target_subdir': 'augmented',
              'aug_metadata_file_name': 'augmented.csv',
              'val_results_file_name': 'validation_results.csv',
              'py_seed': 44,
              'np_seed': 43,
              'tf_seed': 42}

    params['checkpoints_path'] = params['checkpoints_dir'] + '/weights.{epoch:05d}.hdf5'
    params['augm_target_dir'] = params['dataset_root'] + '/' + params['augm_target_subdir']
    params['image_size'] = params['image_shape'][:2]

    with open(params['val_results_file_name'], 'a') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(
            ['Timestamp', 'Elapsed (sec)'] + [param_name for param_name in sorted(params.keys())] + ['F1', 'Precision',
                                                                                                     'Recall',
                                                                                                     'ROC AUC',
                                                                                                     'AP',
                                                                                                     'Loss'])
    tpe_algorithm = tpe.suggest
    trials_path = Path('trials.pickle')
    max_iter = 10

    if trials_path.exists():
        with open(trials_path, 'rb') as pickle_file:
            bayes_trials = pickle.load(pickle_file)
        print('Loaded status of hyper-parameters tuning from file {}, resuming since iteration {}.'.format(
            trials_path,
            len(
                bayes_trials) + 1))
    else:
        bayes_trials = Trials()

    for i in range(max_iter):
        print('Trial no.', len(bayes_trials) + 1)
        best = fmin(fn=run_experiment,
                    space=params,
                    algo=tpe.suggest,
                    max_evals=len(bayes_trials) + 1,
                    trials=bayes_trials,
                    show_progressbar=False)
        with open(trials_path, 'wb') as pickle_file:
            pickle.dump(bayes_trials, pickle_file, pickle.HIGHEST_PROTOCOL)

    print(bayes_trials.best_trial)
    input('All done. Press [Enter] to end.')
