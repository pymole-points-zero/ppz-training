import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess
import json
import os

CWD = os.getcwd()


class TrainingServer(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        # read json config
        config = self.rfile.read(content_length)
        print('Starting training with config: ', config)
        config = json.loads(config)

        # create config file
        with open('configs/config.json', 'w') as f:
            json.dump(config, f)

        # start training process
        args = [
            'python',
            os.path.join(CWD, 'train.py'),
            '--config', os.path.join(CWD, 'configs/config.json')
        ]
        subprocess.Popen(' '.join(args), shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()

    server_address = (args.listen, args.port)
    training_server = HTTPServer(server_address, TrainingServer)

    print(f"Starting training server on {args.listen}:{args.port}")
    training_server.serve_forever()
