import numpy as np
import glob
import os
import sys
import random
import argparse
import json
import datetime
import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import gzip
import io
import tempfile
import tarfile
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import tensorflow as tf


retry_strategy = Retry(
    total=5,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["GET", "POST"],
    backoff_factor=10
)

adapter = HTTPAdapter(max_retries=retry_strategy)


def prepare_session():
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    return s


def upload(model_path, url, best_sha, training_run_id, params):
    with open(model_path, 'rb') as f:
        network = io.BytesIO(gzip.compress(f.read()))

    files = {'network': network}
    params['prev_delta_sha'] = best_sha
    params['training_run_id'] = training_run_id

    session = prepare_session()

    try:
        response = session.post(url, data=params, files=files)
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
    examples = []
    for chunk_path in latest_chunks:
        with tarfile.open(chunk_path, 'r:gz') as tar_chunk:
            for game in tar_chunk:
                gziped_examples = tar_chunk.extractfile(game)
                example = gzip.decompress(gziped_examples.read())
                byte_example = io.BytesIO(example)
                example = np.load(byte_example, allow_pickle=True)
                examples.append(example)

    positions = np.concatenate(examples)
    np.random.shuffle(positions)
    print(len(positions), 'positions')

    return positions


def get_all_chunks(path):
    return glob.glob(os.path.join(path, '*.tar.gz'))


def get_latest_chunks(path, pool_size, num_chunks, allow_less):
    chunk_paths = get_all_chunks(path)
    print('Found', len(chunk_paths), 'chunks at all')

    if len(chunk_paths) < pool_size:
        if not allow_less:
            print(f'Not enough chunks {len(chunk_paths)}/{pool_size}')
            sys.exit(1)

    chunk_paths.sort(key=os.path.getmtime, reverse=True)

    pool = chunk_paths[:pool_size]
    print(f'Constructed pool. {len(pool)} of {pool_size}')

    print('First game generated at ', datetime.datetime.fromtimestamp(os.path.getmtime(pool[-1])))
    print('Last game generated at ', datetime.datetime.fromtimestamp(os.path.getmtime(pool[0])))

    # get random chunks from pool
    random.shuffle(chunk_paths)
    chunk_paths = pool[:num_chunks]

    print(f"Randomly selected {len(chunk_paths)} chunks from pool")

    return chunk_paths


def download_network(url, model_path, sha):
    if os.path.exists(model_path):
        print(sha, 'is cached')
        return

    print('Downloading network', sha)
    session = prepare_session()

    try:
        response = session.get(url, json={'sha': sha})
        with open(model_path, 'wb') as f:
            f.write(response.content)
    except:
        raise Exception


def download_chunks(list_url, download_url, input_path):
    print('Downloading chunks')
    # get latest chunks from server
    session = prepare_session()

    response = session.get(list_url)
    server_latest_chunks = response.json()
    # get missing files
    all_local_chunks = map(os.path.basename, get_all_chunks(input_path))
    missing_chunks = list(set(server_latest_chunks) - set(all_local_chunks))
    print(f'Missing {len(missing_chunks)} chunks')

    # download missing files
    if missing_chunks:
        response = requests.get(download_url, json=missing_chunks)
        missing_chunks_file = io.BytesIO(response.content)

        with tarfile.open(fileobj=missing_chunks_file, mode='r:gz') as tar:
            tar.extractall(input_path)


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    server_conf = config.get('server', None)

    if server_conf is not None:
        if 'examples' in server_conf:
            config['input_path'] = os.path.join('examples', str(server_conf['training_run_id']))
            os.makedirs(config['input_path'], exist_ok=True)

            download_chunks(server_conf['examples']['list_url'], server_conf['examples']['download_url'],
                            config['input_path'])

        if 'network' in server_conf:
            sha = server_conf['best_sha']
            config['model_input'] = os.path.join('networks', sha+'.gz')
            if not os.path.exists('networks'):
                os.mkdir('networks')

            download_network(server_conf['network']['url'], config['model_input'], sha)

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
    if server_conf is not None and 'upload' in server_conf:
        if 'model_output' in config:
            upload(config['model_output'], server_conf['upload']['url'],
                   server_conf['best_sha'], server_conf['training_run_id'],
                   server_conf['upload']['params'])
        else:
            temp_model_path = os.path.join(os.getcwd(), 'temp_model.h5')
            model.save(temp_model_path, save_format='h5', include_optimizer=False)

            upload(temp_model_path, server_conf['upload']['url'],
                   server_conf['best_sha'], server_conf['training_run_id'],
                   server_conf['upload']['params'])

            os.remove(temp_model_path)


# TODO logger
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO inside program it should be config_path
    parser.add_argument('--config', type=str)

    gpus = tf.config.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main(parser.parse_args())