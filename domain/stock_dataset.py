import datetime
import json
import os

import torch
from torch.utils.data.dataset import Dataset
import hashlib
from configs.config import CONFIG
from domain.stock_api import StockAPI


class StockDataset(Dataset):
    CACHE_FOLDER = "./data_cache"

    def __init__(
        self, input_days, keep_columns, output_days=1, before_date: str = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_with_detail = True
        self.data_date = []
        self._stock_api = StockAPI()
        self.selected_stocks = CONFIG["STOCK_CODE_LIST"]

        self._input_days = input_days
        self.keep_columns = keep_columns
        self.output_days = output_days

        self.before_date = before_date

        self._data = self._get_data()

    def _get_data(self):
        # check cache folder
        current_date = datetime.datetime.now().date().strftime("%Y%m%d")

        if self.before_date is not None:
            current_date = self.before_date

        cache_file_name = current_date
        sorted_stock_code_list = sorted(CONFIG["STOCK_CODE_LIST"])
        # hash stock code list
        cache_file_name += "_" + hashlib.md5(''.join(sorted_stock_code_list).encode()).hexdigest()

        if os.path.exists(os.path.join(self.CACHE_FOLDER, f"{cache_file_name}.json")):
            with open(
                os.path.join(self.CACHE_FOLDER, f"{cache_file_name}.json"), "r"
            ) as f:
                stock_data_list = json.load(f)

        else:
            # get data from api
            stock_data_list = []
            for stock_code in CONFIG["STOCK_CODE_LIST"]:
                success, data = self._stock_api.get_info(stock_code)
                if not success:
                    raise Exception(
                        f"Cannot get data from api. Stock code: {stock_code}"
                    )
                stock_data_list.append({"stock_code": stock_code, "data": data})
            # save data to cache
            with open(
                os.path.join(self.CACHE_FOLDER, f"{cache_file_name}.json"), "w"
            ) as f:
                json.dump(stock_data_list, f)

        for stock_data in stock_data_list:
            for x in stock_data["data"]:
                x["date"] = datetime.datetime.fromtimestamp(int(x["date"]))

        for stock_data in stock_data_list:
            data = stock_data["data"]
            data = sorted(data, key=lambda x: x["date"])
            data = self._filter_duplicate(data)
            data = self._filter_zero(data)

            # remove data after before_date
            if self.before_date is not None:
                data_type_before_date = datetime.datetime.strptime(
                    self.before_date, "%Y%m%d"
                ).date()
                data = [x for x in data if x["date"].date() <= data_type_before_date]

            # keep only the columns we need
            data_date = [x["date"] for x in data]
            # convert back to timestamp
            data_date = [x.timestamp() for x in data_date]
            stock_data["data_date"] = data_date

            stock_data["data"] = [
                torch.Tensor([x[column] for column in self.keep_columns]).to(
                    self.device
                )
                for x in data
            ]
            stock_data["data"] = torch.stack(stock_data["data"]).to(self.device)

        return stock_data_list

    def __getitem__(self, index):
        cur_len = 0
        for stock_data in self._data:
            if stock_data["stock_code"] in self.selected_stocks:
                if index >= self._len_of_data(stock_data["data"]) + cur_len:
                    cur_len += self._len_of_data(stock_data["data"])
                    continue
                else:
                    index -= cur_len
                    x = stock_data["data"][index : index + self._input_days]
                    y = stock_data["data"][
                        index
                        + self._input_days : index
                        + self._input_days
                        + self.output_days
                    ]
                    if self.return_with_detail:
                        return (
                            x,
                            y,
                            {
                                "stock_code": stock_data["stock_code"],
                                "data_date": stock_data["data_date"][
                                    index : index + self._input_days + self.output_days
                                ],
                            },
                        )
                    else:
                        return (x, y)

    def set_return_with_detail(self, return_with_detail_flag):
        self.return_with_detail = return_with_detail_flag

    def get_return_with_detail(self):
        return self.return_with_detail

    def set_selected_stocks(self, selected_stocks):
        self.selected_stocks = selected_stocks

    def _len_of_data(self, data):
        return len(data) - (self._input_days + self.output_days) + 1

    def __len__(self):
        total_len = 0
        for idx, stock_data in enumerate(self._data):
            if stock_data["stock_code"] in self.selected_stocks:
                total_len += self._len_of_data(stock_data["data"])

        return total_len

    def _filter_duplicate(self, data):
        same_day_data = []
        for i in range(1, len(data)):
            if data[i]["date"].date() == data[i - 1]["date"].date():
                same_day_data.append(data[i - 1])
        for x in same_day_data:
            data.remove(x)
        return data

    def _filter_zero(self, data):
        zero_data = []
        for x in data:
            if x["open"] == 0 or x["high"] == 0 or x["low"] == 0 or x["close"] == 0:
                zero_data.append(x)
        for x in zero_data:
            data.remove(x)
        return data

    def get_all_data(self):
        return self._data
