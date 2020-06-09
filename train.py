import numpy as np
import glob
import os
import sys
import random
import argparse
import json
import requests
import gzip
import io
import tempfile
from tensorflow import keras
from tensorflow.keras.optimizers import SGD


def upload(model_path, url, params):
    with open(model_path, 'rb') as f:
        network = io.BytesIO(gzip.compress(f.read()))

    files = {'network': network}

    try:
        response = requests.post(url, data=params, files=files)
        print(response.json())
    except Exception as e:
        print(e)
    finally:
        network.close()


def train(model, examples, batch_size):
    model.fit(
        x=examples['field'],
        y=[examples['policy'], examples['value']],
        batch_size=batch_size,
        epochs=1,
    )


def train_test_split(examples, train_ratio):
    np.random.shuffle(examples)
    split_index = int(len(examples) * train_ratio)
    return examples[:split_index], examples[split_index:]


def load_model(tmp_dir, model_path):
    with gzip.open(model_path, 'rb') as f:
        with tempfile.NamedTemporaryFile('wb', dir=tmp_dir.name) as tmp_file:
            tmp_file.write(f.read())
            model = keras.models.load_model(tmp_file.name)

    return model

def get_examples(latest_chunks):
    examples = []
    for path in latest_chunks:
        with gzip.open(path, 'rb') as f:
            example = np.load(f, allow_pickle=True)
            examples.append(example)

    # TODO shuffle this
    examples = np.concatenate(examples)
    print(len(examples), 'positions')

    return examples


def get_all_chunks(path):
    return [os.path.join(path, filename) for filename in glob.glob(os.path.join(path, '*.gz'))]


def get_latest_chunks(path, num_chunks, allow_less):
    chunk_paths = get_all_chunks(path)

    if len(chunk_paths) < num_chunks:
        if allow_less:
            print(f'Got {len(chunk_paths)} of {num_chunks}')
        else:
            print(f'Not enough chunks {len(chunk_paths)}/{num_chunks}')
            sys.exit(1)

    chunk_paths.sort(key=os.path.getmtime, reverse=True)

    chunk_paths = chunk_paths[:num_chunks]
    print('First chunk generated at ', os.path.getmtime(chunk_paths[0]))
    print('Last chunk generated at ', os.path.getmtime(chunk_paths[-1]))

    random.shuffle(chunk_paths)
    return chunk_paths


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    latest_chunks = get_latest_chunks(config['input_path'], config['num_chunks'], config['allow_less'])
    examples = get_examples(latest_chunks)
    # train, test = train_test_split(examples, config.train_ratio)

    tmp_dir = tempfile.TemporaryDirectory()

    model = load_model(tmp_dir, config['model_input'])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=SGD(
        learning_rate=config['lr'], momentum=config['momentum']))

    train(model, examples, config['batch_size'])

    if 'model_output' in config:
        model.save(config['model_output'], save_format='h5', include_optimizer=False)

    # TODO separate upload and train into different files
    if 'upload' in config:
        if 'model_output' in config:
            upload(config['model_output'], config['upload']['url'], config['upload']['params'])
        else:
            temp_model_path = os.path.join(os.getcwd(), 'temp_model.h5')
            model.save(temp_model_path, save_format='h5', include_optimizer=False)

            upload(temp_model_path, config['upload']['url'], config['upload']['params'])

            os.remove(temp_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO inside program it should be config_path
    parser.add_argument('--config', type=str)

    main(parser.parse_args())