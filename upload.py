from train import upload
import json

with open('configs/config.json', 'r') as f:
    config = json.load(f)

other_params = {
    'filters': 256,
    'blocks': 5,
    'field_width': 10,
    'field_height': 10,
    'training_run_id': 4
}

upload('model.h5', config['upload']['url'], other_params)