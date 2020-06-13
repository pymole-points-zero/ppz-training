import numpy as np
import glob
import os
import sys
import random
import argparse
import json
import datetime
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


def train(model, positions, batch_size):
    model.fit(
        x=positions['field'],
        y=[positions['policy'], positions['value']],
        batch_size=batch_size,
        epochs=1,
    )


def train_test_split(positions, train_ratio):
    split_index = int(len(positions) * train_ratio)
    return positions[:split_index], positions[split_index:]


def load_model(tmp_dir, model_path):
    with gzip.open(model_path, 'rb') as f:
        with tempfile.NamedTemporaryFile('wb', dir=tmp_dir.name) as tmp_file:
            tmp_file.write(f.read())
            model = keras.models.load_model(tmp_file.name)

    return model


def get_positions(latest_chunks):
    positions = []
    for path in latest_chunks:
        with gzip.open(path, 'rb') as f:
            chunks = np.load(f, allow_pickle=True)
            positions.append(chunks)

    positions = np.concatenate(positions)
    np.random.shuffle(positions)
    print(len(positions), 'positions')

    return positions


def get_all_chunks(path):
    return [os.path.join(path, filename) for filename in glob.glob(os.path.join(path, '*.gz'))]


def get_latest_chunks(path, pool_size, num_chunks, allow_less):
    chunk_paths = get_all_chunks(path)
    print(len(chunk_paths), 'games at all')

    if len(chunk_paths) < pool_size:
        if not allow_less:
            print(f'Not enough chunks {len(chunk_paths)}/{pool_size}')
            sys.exit(1)

    chunk_paths.sort(key=os.path.getmtime, reverse=True)

    pool = chunk_paths[:pool_size]
    print(f'Constructed pool. {len(pool)} of {pool_size}')

    print('First game generated at ', datetime.datetime.fromtimestamp(os.path.getmtime(pool[0])))
    print('Last game generated at ', datetime.datetime.fromtimestamp(os.path.getmtime(pool[-1])))

    # get random games from pool
    random.shuffle(chunk_paths)
    chunk_paths = pool[:num_chunks]

    print(f"Randomly selected {chunk_paths} games from pool")

    return chunk_paths


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    latest_chunks = get_latest_chunks(config['input_path'], config['pool_size'],
                                      config['num_chunks'], config['allow_less'])
    positions = get_positions(latest_chunks)
    # train, test = train_test_split(examples, config.train_ratio)

    tmp_dir = tempfile.TemporaryDirectory()

    model = load_model(tmp_dir, config['model_input'])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=SGD(
        learning_rate=config['lr'], momentum=config['momentum']))

    train(model, positions, config['batch_size'])

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