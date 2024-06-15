import torch


class PredictRepeater:
    def __init__(self, model=None):
        self.model = model

    def set_model(self, model):
        self.model = model

    def repeatly_predict(self, input_data, repeat_times):
        predict_result = []
        for i in range(repeat_times):
            output = self.model(input_data)
            output = output[:, -1, :]
            predict_result.append(output)
            print(input_data.shape, output.shape)
            input_data = torch.cat([input_data, output], dim=1)
            input_data = input_data[:, 1:, :]

        return predict_result
