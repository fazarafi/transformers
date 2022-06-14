import argparse
import os


past_versions_testing = {
    "torch": {
        "1.10": {
            "torch": "1.10.2",
            "torchvision": "0.11.3",
            "torchaudio": "0.10.2",
            "python": 3.9,
            "cuda": "cu113",
            "install": "python3 -m pip install --no-cache-dir -U torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu113",
        },
        # torchaudio < 0.10 has no CUDA-enabled binary distributions
        "1.9": {
            "torch": "1.9.1",
            "torchvision": "0.10.1",
            "torchaudio": "0.9.1",
            "python": 3.9,
            "cuda": "cu111",
            "install": "python3 -m pip install --no-cache-dir -U torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 --extra-index-url https://download.pytorch.org/whl/cu111",
        },
        "1.8": {
            "torch": "1.8.1",
            "torchvision": "0.9.1",
            "torchaudio": "0.8.1",
            "python": 3.9,
            "cuda": "cu111",
            "install": "python3 -m pip install --no-cache-dir -U torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 --extra-index-url https://download.pytorch.org/whl/cu111",
        },
        "1.7": {
            "torch": "1.7.1",
            "torchvision": "0.8.2",
            "torchaudio": "0.7.2",
            "python": 3.9,
            "cuda": "cu110",
            "install": "python3 -m pip install --no-cache-dir -U torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 --extra-index-url https://download.pytorch.org/whl/cu110",
        },
        "1.6": {
            "torch": "1.6.0",
            "torchvision": "0.7.0",
            "torchaudio": "0.6.0",
            "python": 3.8,
            "cuda": "cu101",
            "install": "python3 -m pip install --no-cache-dir -U torch==1.6.0 torchvision==0.7.0 torchaudio==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu101",
        },
        "1.5": {
            "torch": "1.5.1",
            "torchvision": "0.6.1",
            "torchaudio": "0.5.1",
            "python": 3.8,
            "cuda": "cu101",
            "install": "python3 -m pip install --no-cache-dir -U torch==1.5.1 torchvision==0.6.1 torchaudio==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu101",
        },
        "1.4": {
            "torch": "1.4.0",
            "torchvision": "0.5.0",
            "torchaudio": "0.4.0",
            "python": 3.8,
            "cuda": "cu100",
            "install": "python3 -m pip install --no-cache-dir -U torch==1.4.0 torchvision==0.5.0 torchaudio==0.4.0 --extra-index-url https://download.pytorch.org/whl/cu100",
        },
        # need python 3.7
        # "1.3": {
        #     "torch": "1.3.1",
        #     "torchvision": "0.4.2",
        #     "torchaudio": None,
        #     "python": 3.7,
        #     "cuda": "cu100",
        #     "docker-base": "10.0-cudnn7-devel-ubuntu18.04",
        #     "install": "python3 -m pip install --no-cache-dir -U torch==1.3.1 torchvision==0.4.2 torchaudio==0.4.0 --extra-index-url https://download.pytorch.org/whl/cu100",
        # },
    },
    "tensorflow": {
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Choose the framework and version to install")
    parser.add_argument("--framework", help="The framework to install. Should be `torch` or `tensorflow`", type=str)
    parser.add_argument("--version", help="The version of the framework to install.", type=str)
    args = parser.parse_args()

    info = past_versions_testing[args.framework][args.version]

    os.system(f'echo "export INSTALL_CMD=\'{info["install"]}\'" >> ~/.profile')
    os.system(f'echo "export CUDA=\'{info["cuda"]}\'" >> ~/.profile')

    print(f'echo "export INSTALL_CMD=\'{info["install"]}\'" >> ~/.profile')
    print(f'echo "export CUDA=\'{info["cuda"]}\'" >> ~/.profile')
