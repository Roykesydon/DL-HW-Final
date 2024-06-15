import argparse
import datetime
import math
import os
import random
import time

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from config import CONFIG
from domain.figure_generator import FigureGenerator
from domain.loss.rmse import RMSELoss
from domain.models.stock_prediction.mamba import MambaPredictionModel
from domain.models.stock_prediction.transformer import \
    TransformerPredictionModel
from domain.preprocessor import Preprocessor
from domain.stock_dataset import StockDataset


def validation(model, data_loader, criterion):
    model.eval()
    val_loss = 0
    total_data = 0
    with torch.no_grad():
        for x, y in data_loader:
            # preprocess
            x = preprocessor.z_score_normalize(x, "train")
            y = preprocessor.z_score_normalize(y, "train")

            output = model(x)

            # denormalize
            output = preprocessor.z_score_denormalize(output, "train")
            y = preprocessor.z_score_denormalize(y, "train")

            loss = RMSELoss()(output[:, -1, :], y)
            val_loss += loss.item() * len(x)
            total_data += len(x)
    return val_loss / total_data


if __name__ == "__main__":
    random_seed = 256
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    msg = "This is a stock prediction validation script"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        "--model",
        type=str,
        help="select model. options: transformer, mamba",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        help="path to the weight file",
    )
    args = parser.parse_args()

    """
    File Path
    """
    TRAINING_DATA_PATH = CONFIG["TRAINING_DATA_PATH"]
    NEWEST_DATA_PATH = CONFIG["NEWEST_DATA_PATH"]
    SAVE_RESULTS_FOLDER = CONFIG["SAVE_RESULTS_FOLDER"]
    """
    Transformer model for stock prediction
    """
    TRANSFORMER_HYPER = {
        "D_MODEL": 56,
        "N_HEAD": 8,
        "NUM_ENCODER_LAYERS": 2,
        "DROPOUT": 0,
        "NUM_FEATURES": 4,
        "LEARNING_RATE": 8e-4,
    }

    """
    Mamba model for stock prediction
    """
    MAMBA_HYPER = {
        "D_MODEL": 16,
        "D_STATE": 16,
        "D_CONV": 4,
        "EXPAND": 2,
        "NUM_FEATURES": 4,
        "LEARNING_RATE": 1e-3,
    }

    """
    Model setting
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    figure_generator = FigureGenerator()
    preprocessor = Preprocessor()

    transformer_prediction_model = TransformerPredictionModel(
        d_model=TRANSFORMER_HYPER["D_MODEL"],
        nhead=TRANSFORMER_HYPER["N_HEAD"],
        num_encoder_layers=TRANSFORMER_HYPER["NUM_ENCODER_LAYERS"],
        dropout=TRANSFORMER_HYPER["DROPOUT"],
        num_features=TRANSFORMER_HYPER["NUM_FEATURES"],
    ).to(device)

    mamba_prediction_model = MambaPredictionModel(
        d_model=MAMBA_HYPER["D_MODEL"],
        d_state=MAMBA_HYPER["D_STATE"],
        d_conv=MAMBA_HYPER["D_CONV"],
        expand=MAMBA_HYPER["EXPAND"],
        num_features=MAMBA_HYPER["NUM_FEATURES"],
    ).to(device)

    model_dict = {
        "transformer": {
            "hyperparameters": TRANSFORMER_HYPER,
            "model": transformer_prediction_model,
        },
        "mamba": {
            "hyperparameters": MAMBA_HYPER,
            "model": mamba_prediction_model,
        },
    }

    model_selection = args.model

    model = model_dict[model_selection]["model"]
    learning_rate = model_dict[model_selection]["hyperparameters"]["LEARNING_RATE"]

    """
    Training hyperparameters
    """
    EPOCHS = 500
    BATCH_SIZE = 64
    criterion = RMSELoss()
    # criterion = MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    EARLY_STOPPING = 10

    """
    Dataset and DataLoader
    """
    dataset = StockDataset(
        data_path=TRAINING_DATA_PATH,
        input_days=30,
        keep_columns=["open", "high", "low", "close"],
    )
    print(f"Dataset length: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    valid_size = int(0.2 / 0.8 * len(train_dataset))
    train_size = len(train_dataset) - valid_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if args.model == "transformer":
        if os.path.exists(args.weight_path):
            transformer_prediction_model.load_state_dict(torch.load(args.weight_path))
        else:
            raise ValueError("Invalid weight path")

    elif args.model == "mamba":
        if os.path.exists(args.weight_path):
            mamba_prediction_model.load_state_dict(torch.load(args.weight_path))
        else:
            raise ValueError("Invalid weight path")

    else:
        raise ValueError(
            "Invalid model selection. Please select either 'transformer' or 'mamba'"
        )

    """
    Model Structure
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {pytorch_total_params}")

    """
    Testing
    """
    model.eval()
    weighted_loss = 0
    total_data = 0
    preprocessor.calc_mean_and_std(train_dataset.dataset, "train")
    start_time = time.time()
    test_loss = validation(model, test_loader, criterion)
    print(f"Test loss: {test_loss}")
    print(f"Inference test dataset cosuming time(s): {time.time() - start_time}")

    """
    Backtesting
    """
    while True:
        # 輸入 2024 的其中一天開市日
        # 最新的資料更新到 3/11
        print("-" * 10)
        print("Please input a date (yyyy-mm-dd) in 2024")
        print("The newest data is updated to 3/11")
        input_date = input("Date (yyyy-mm-dd): ")
        # input: 2024/03/11
        dataset = StockDataset(
            data_path=NEWEST_DATA_PATH,
            input_days=30,
            keep_columns=["open", "high", "low", "close"],
        )

        # parse input date
        date = datetime.datetime.strptime(input_date, "%Y-%m-%d")

        # check if the date is in the dataset
        test_index = -1
        dataset.set_return_with_date(True)
        for i, data in enumerate(dataset):
            x, y, date_list = data
            if date_list[-1].date() == date.date():
                test_index = i
                break

        if test_index == -1:
            print("The date is not in the dataset")
            continue

        break

    # get the data
    x, y, date_list = dataset[test_index]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    # preprocess
    x = preprocessor.z_score_normalize(x, "train")
    y = preprocessor.z_score_normalize(y, "train")

    # predict
    output = model(x)

    # denormalize
    x = preprocessor.z_score_denormalize(x, "train")
    y = preprocessor.z_score_denormalize(y, "train")
    output = preprocessor.z_score_denormalize(output, "train")

    # loss
    loss = criterion(output[:, -1, :], y)
    print(f"RMSE Loss: {loss.item()}")

    # plot
    save_path = f"{SAVE_RESULTS_FOLDER}/backtesting_{model_selection}_{input_date}_{int(datetime.datetime.now().timestamp())}.png"
    figure_generator.save_test_result_figure(
        f"{save_path}",
        {
            "x": x[0],
            "y": y[0],
            "predict": output[0][-1],
        },
    )

    print(f"Backtesting figure is saved at {save_path}")
