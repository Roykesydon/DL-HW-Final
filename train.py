import argparse
import datetime
import math
import os
import random
from copy import deepcopy

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from configs.config import CONFIG
from domain.figure_generator import FigureGenerator
from domain.loss.rmse import RMSELoss
from domain.models.stock_prediction.mamba import MambaPredictionModel
from domain.models.stock_prediction.transformer import TransformerPredictionModel
from domain.preprocessor import Preprocessor
from domain.stock_dataset import StockDataset


def validation(model, data_loader, criterion):
    model.eval()
    val_loss = 0
    total_data = 0
    with torch.no_grad():
        for x, y, detail in data_loader:
            # preprocess
            for i in range(len(detail)):
                x[i] = preprocessor.z_score_normalize(x[i], detail["stock_code"][i])
                y[i] = preprocessor.z_score_normalize(y[i], detail["stock_code"][i])

            output = model(x)

            # # denormalize
            # for i in range(len(detail)):
            #     output[i] = preprocessor.z_score_denormalize(
            #         output[i], detail["stock_code"][i]
            #     )
            #     y[i] = preprocessor.z_score_denormalize(y[i], detail["stock_code"][i])

            # loss = criterion(output[:, -1, :, :].squeeze(dim=1), y)
            loss = criterion(output[:, -1, :, :].squeeze(dim=1), y[:, -1, :])

            val_loss += loss.item() * len(x)
            total_data += len(x)
    return val_loss / total_data


if __name__ == "__main__":
    msg = "This is a stock prediction training script"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        "--model",
        type=str,
        help="select model. options: transformer, mamba",
    )

    parser.add_argument(
        "--random",
        help="don't set random seed",
        action="store_true",
    )
    args = parser.parse_args()

    """
    Random seed
    """
    if not args.random:
        random_seed = 256
        torch.manual_seed(random_seed)
        random.seed(random_seed)

    """
    File Path
    """
    TRAINING_DATA_PATH = CONFIG["TRAINING_DATA_PATH"]
    SAVE_RESULTS_FOLDER = CONFIG["SAVE_RESULTS_FOLDER"]
    """
    Common hyperparameters
    """
    INPUT_DAYS = 90
    OUTPUT_DAYS = 10
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
        "D_MODEL": 6, # 6
        "D_STATE": 6, # 6
        "D_CONV": 4, # 4 
        "EXPAND": 8, # 8
        "NUM_FEATURES": 4,
        "LEARNING_RATE": 1e-3, # 1e-3
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
        input_days=INPUT_DAYS,
        # output_days=OUTPUT_DAYS,
        output_days=1,
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
    EPOCHS = 2000
    BATCH_SIZE = 64
    criterion = RMSELoss()
    # criterion = MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    EARLY_STOPPING = 20
    BEFORE_DATE = "20240301"

    """
    Dataset and DataLoader
    """
    dataset = StockDataset(
        input_days=INPUT_DAYS,
        keep_columns=["open", "high", "low", "close"],
        output_days=OUTPUT_DAYS,
        before_date=BEFORE_DATE,
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

    """
    Training
    """
    best_result = {
        "epoch": 0,
        "train_loss": math.inf,
        "valid_loss": math.inf,
        "test_loss": math.inf,
    }
    figure_generator.init_epoch_record()
    preprocessor.calc_mean_and_std(train_dataset.dataset, CONFIG["STOCK_CODE_LIST"])
    for epoch in range(EPOCHS):
        # early stopping
        if epoch - best_result["epoch"] > EARLY_STOPPING:
            print("Early stopping")
            print(f"Best valid result:")
            print(f"train_loss: {best_result['train_loss']}")
            print(f"valid_loss: {best_result['valid_loss']}")
            print(f"test_loss: {best_result['test_loss']}")

            # make folder
            save_folder = f"{SAVE_RESULTS_FOLDER}/{model.__class__.__name__}_"
            save_folder += f"{best_result['valid_loss']:.3f}"
            save_folder += f"_{int(datetime.datetime.now().timestamp())}"
            os.makedirs(f"{save_folder}", exist_ok=True)

            # save epoch figure
            figure_generator.save_epoch_record_figure(f"{save_folder}/epoch_figure.png")

            # save testing figure
            # randomly choose 5 batch from test_loader
            test_data = random.sample(list(test_loader), 5)

            for idx, (x, y, detail) in enumerate(test_data):
                # preprocess
                for i in range(len(detail)):
                    x[i] = preprocessor.z_score_normalize(x[i], detail["stock_code"][i])
                    y[i] = preprocessor.z_score_normalize(y[i], detail["stock_code"][i])

                output = model(x)

                # denormalize
                for i in range(len(detail)):
                    x[i] = preprocessor.z_score_denormalize(
                        x[i], detail["stock_code"][i]
                    )
                    output[i] = preprocessor.z_score_denormalize(
                        output[i], detail["stock_code"][i]
                    )
                    y[i] = preprocessor.z_score_denormalize(
                        y[i], detail["stock_code"][i]
                    )

                # linear interpolate
                # expand to OUTPUT_DAYS
                # output shape torch.Size([64, 30, 1, 4])
                new_output = torch.zeros((output.shape[0], OUTPUT_DAYS, 4))
                for i in range(output.shape[0]):
                    for j in range(4):
                        new_output[i, :, j] = torch.linspace(
                            x[i, -1, j], output[i, -1, 0, j], OUTPUT_DAYS
                        )[-OUTPUT_DAYS:]

                output = new_output

                current_batch_size = x.size(0)
                random_idx = random.randint(0, current_batch_size - 1)

                figure_generator.save_test_result_figure(
                    f"{save_folder}/test_result_{idx}.png",
                    {
                        "x": x[random_idx],
                        "y": y[random_idx],
                        "predict": output[random_idx],
                    },
                )

            # save model
            torch.save(
                best_result["model"],
                f"{save_folder}/weight.pth",
            )

            # save hyperparameters to txt
            with open(
                f"{save_folder}/info.txt",
                "w",
            ) as f:
                f.write(f"Model: {model.__class__.__name__}\n")
                f.write(f"Befor Date: {BEFORE_DATE}\n")
                f.write(f"Best valid result:\n")
                f.write(f"train_loss: {best_result['train_loss']}\n")
                f.write(f"valid_loss: {best_result['valid_loss']}\n")
                f.write(f"test_loss: {best_result['test_loss']}\n")
                f.write(f"Learning Rate: {learning_rate}\n")
                f.write(f"Epochs: {EPOCHS}\n")
                f.write(f"Batch Size: {BATCH_SIZE}\n")
                f.write(f"Input Days: {INPUT_DAYS}\n")
                f.write(f"Output Days: {OUTPUT_DAYS}\n")
                f.write(f"{model.get_hyperparameters_str()}\n")
                f.write(f"mean_and_std:\n")
                f.write(f"{preprocessor.get_mean_and_std()}\n")

            print(f"Result saved at folder: {save_folder}")
            print("Tree structure of folder:")
            os.system(f"tree {save_folder}")

            break

        model.train()

        for x, y, detail in train_loader:

            for i in range(len(detail)):
                x[i] = preprocessor.z_score_normalize(x[i], detail["stock_code"][i])
                y[i] = preprocessor.z_score_normalize(y[i], detail["stock_code"][i])

            optimizer.zero_grad()
            output = model(x)
            # loss = criterion(output[:, -1, :, :].squeeze(dim=1), y)
            # loss1 = criterion(output[:, -1, 0, :].squeeze(dim=1), y[:, 0, :])
            loss = criterion(output[:, -1, :, :].squeeze(dim=1), y[:, -1, :])

            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print("=" * 10)
            print(f"Epoch: {epoch}")
            train_loss = validation(model, train_loader, criterion)
            valid_loss = validation(model, valid_loader, criterion)
            print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}")
            test_loss = validation(model, test_loader, criterion)
            print(f"Test Loss: {test_loss}")
            test_loss = validation(model, test_loader, criterion)

            if valid_loss < best_result["valid_loss"]:
                best_result["epoch"] = epoch
                best_result["train_loss"] = train_loss
                best_result["valid_loss"] = valid_loss
                best_result["test_loss"] = test_loss
                best_result["model"] = deepcopy(model)

            figure_generator.add_epoch_record(epoch, train_loss, valid_loss, test_loss)

            print("=" * 10)
