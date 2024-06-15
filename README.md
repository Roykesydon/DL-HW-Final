## NCKU-DL-Final

### Prepare Environment

#### Option 1: Use Docker
1. Build the docker image and run the container
    ```bash
    docker build -t ncku-dl-final-p76124168 . --no-cache
    ```
    
2. Run the docker container
    ```bash
    docker run -d -it --name ncku-dl-final-p76124168 --shm-size 1G --gpus device=0 -v $(pwd):/workspace ncku-dl-final-p76124168 bash
    ```

3. After running the container, you can use the following command to enter the container.
    ```bash
    docker exec -it ncku-dl-final-p76124168 bash
    ```

##### Remove
1. Stop the container
    ```bash
    docker stop ncku-dl-final-p76124168
    ```
2. Remove the container
    ```bash
    docker rm ncku-dl-final-p76124168
    ```
3. Remove the image
    ```bash
    docker rmi ncku-dl-final-p76124168
    ```

#### Option 2: Install the environment manually
1. Prepare Python with version 3.8
2. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

### Scripts

#### start.sh
This script is for TAs to run the training and backtesting scripts.
```bash
bash start.sh
```

#### Download Dataser
This script is for downloading the training and newest dataset from the API.
```bash
python download_dataset.py
```

#### Training
You can use this script to train mamba or transformer model.

1. Make sure you have the dataset file in the path you set in the config.py

2. Run the training script.
    You need to specify the model you want to train in the config.py.

    You can choose "mamba" or "transformer".
    ```bash
    python train.py --model <model_name> <--random>
    ```

#### Simulation
You can use this script to simulate the trading strategy.

1. Copy information of a training result to `simulate.py`

2. Run the simulation script.
    ```bash
    python simulate.py
    ```

#### Proxy
You can use this script to trade by process

1. Copy information of a training result to `proxy.py`

2. Run the proxy script.
    ```bash
    python proxy.py
    ```