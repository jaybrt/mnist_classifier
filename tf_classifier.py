import os
import tensorflow as tf
import tensorflow_datasets as tfds
from time import perf_counter
import csv


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255., label

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split = ['train', 'test'],
    shuffle_files = True,
    as_supervised = True,
    with_info = True,
)

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


def new_model(activ_func):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation=activ_func),
        tf.keras.layers.Dense(128, activation=activ_func),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    start = perf_counter()

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    end = perf_counter()

    train_time = end-start

    accuracy = model.evaluate(
        x=ds_train,
    )[1]*100

    print(f'training time = {train_time}s')
    print(f'accuracy:{accuracy}%')
    print('--------------------------------------')

    return train_time, accuracy

activ_func = input('activation function?\n')
if activ_func == 'test':
    results = []
    for func in ['relu', 'sigmoid', 'tanh']:
        for i in range(10):
            print(func, i+1)
            train_time, accuracy = new_model(func)
            res = (i+1, func, train_time, accuracy)
            results.append(res)

    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Activation Function', 'Iteration', 'Training Time', 'Accuracy'])
        writer.writerows(results)

else:
    new_model(activ_func)




