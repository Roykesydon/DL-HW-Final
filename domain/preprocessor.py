import torch


class Preprocessor:
    def __init__(self):
        self.z_score = {
            # "<stock code>": {
            #     "mean": None,
            #     "std": None,
            # },
        }

    def calc_mean_and_std(self, dataset, selected_stocks):
        # Handling Subset objects specifically
        if isinstance(dataset, torch.utils.data.Subset):
            data_list = []
            detail_list = []
            for idx in dataset.indices:
                x, y, details = dataset.dataset[idx]
                data = torch.cat((x, y), dim=0)
                data_list.append(data)
                detail_list.append(details)

            data_tensor = torch.stack(data_list)
        else:
            # If not a Subset, assume the dataset itself is a tensor or has the .mean() and .std() methods
            raise NotImplementedError()
            # data_tensor = dataset

        for selected_stock in selected_stocks:
            mean_list = []
            std_list = []
            # iterate all data
            for i in range(len(detail_list)):
                if detail_list[i]["stock_code"] == selected_stock:
                    mean_list.append(data_tensor[i])
                    std_list.append(data_tensor[i])

            mean = torch.stack(mean_list).view(-1, data_tensor.shape[-1]).mean(dim=0)
            std = torch.stack(std_list).view(-1, data_tensor.shape[-1]).std(dim=0)

            self.z_score[selected_stock] = {"mean": mean, "std": std}

    def z_score_normalize(self, tensor, stock_code):
        if stock_code in self.z_score.keys():
            mean = self.z_score[stock_code]["mean"]
            std = self.z_score[stock_code]["std"]
            return (tensor - mean) / std
        else:
            raise ValueError("stock_code does not exist")

    def z_score_denormalize(self, tensor, stock_code):
        if stock_code in self.z_score.keys():
            mean = self.z_score[stock_code]["mean"]
            std = self.z_score[stock_code]["std"]
            return tensor * std + mean
        else:
            raise ValueError("stock_code does not exist")
        
    def get_mean_and_std(self):
        return self.z_score
