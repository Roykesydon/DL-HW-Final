import matplotlib.pyplot as plt
import pandas as pd


class FigureGenerator:
    def __init__(self):
        self._epoch_record = []

    def init_epoch_record(self):
        self._epoch_record = []

    def add_epoch_record(self, epoch, train_loss, val_loss, test_loss):
        self._epoch_record.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            }
        )

    def save_epoch_record_figure(self, output_path):
        df = pd.DataFrame(self._epoch_record)
        df.set_index("epoch", inplace=True)
        df.plot()
        plt.savefig(output_path)
        plt.close()

    def save_test_result_figure(self, output_path, x_y_predict):
        for key in x_y_predict:
            x_y_predict[key] = x_y_predict[key].cpu().detach().numpy()

        # prepend x to y to [31, 4]
        x_y_predict["x"] = x_y_predict["x"].tolist()

        if "y" in x_y_predict.keys() and "predict" in x_y_predict.keys():
            # check if y is 1D
            if len(x_y_predict["y"].shape) == 1:
                actual_list = x_y_predict["x"] + [x_y_predict["y"].tolist()]
                predict_list = x_y_predict["x"] + [x_y_predict["predict"].tolist()]
            elif len(x_y_predict["y"].shape) == 2:
                actual_list = x_y_predict["x"] + x_y_predict["y"].tolist()
                predict_list = x_y_predict["x"] + x_y_predict["predict"].tolist()
        else:
            actual_list = x_y_predict["x"]

        # start figure
        plt.figure(figsize=(10, 5))
        plt.title("Test result")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)

        # plot actual and predict
        columns = ["open", "high", "low", "close"]
        color_list = ["blue", "green", "red", "orange"]
        for idx in range(4):
            cur_actual = [item[idx] for item in actual_list]
            plt.plot(cur_actual, label=f"actual_{columns[idx]}", color=color_list[idx])
            
            if "y" in x_y_predict.keys() and "predict" in x_y_predict.keys():
                cur_predict = [item[idx] for item in predict_list]
                plt.plot(
                    cur_predict,
                    label=f"predict_{columns[idx]}",
                    color=color_list[idx],
                    linestyle="--",
                )

        # legend
        plt.legend()

        # save figure
        plt.savefig(output_path)

        # close figure
        plt.close()
