from typing import Tuple

import requests

from configs.config import CONFIG


class StockAPI:
    def buy(self, stock_code, stock_shares, price) -> bool:
        params = {
            "account": CONFIG["ACCOUNT"],
            "password": CONFIG["PASSWORD"],
            "stock_code": stock_code,
            "stock_shares": stock_shares,
            "stock_price": price,
        }
        url = "http://140.116.86.242:8081/stock/api/v1/buy"
        response = requests.post(url, data=params).json()

        if response["result"] == "success":
            return True
        print(response["status"])
        return False

    def sell(self, stock_code, stock_shares, price) -> bool:
        params = {
            "account": CONFIG["ACCOUNT"],
            "password": CONFIG["PASSWORD"],
            "stock_code": stock_code,
            "stock_shares": stock_shares,
            "stock_price": price,
        }
        url = "http://140.116.86.242:8081/stock/api/v1/sell"
        response = requests.post(url, data=params).json()

        if response["result"] == "success":
            return True
        print(response["status"])
        return False

    def get_info(
        self, stock_code, start_date="20000101", end_date="20290101"
    ) -> Tuple[bool, dict]:
        url = f"http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{end_date}"
        response = requests.get(url).json()

        if response["result"] == "success":
            return True, response["data"]
        return False, {}

    def get_user_stocks(self) -> Tuple[bool, dict]:
        params = {
            "account": CONFIG["ACCOUNT"],
            "password": CONFIG["PASSWORD"],
        }
        url = "http://140.116.86.242:8081/stock/api/v1/get_user_stocks"
        response = requests.post(url, data=params).json()

        if response["result"] == "success":
            return True, response["data"]
        # [{'usid': 4291, 'stock_name': '大飲', 'beginning_price': 7.1, 'shares': 1, 'createtime': 1712129497, 'stock_code_id': '1213', 'user_uid_id': 1124}]
        return False, {}
