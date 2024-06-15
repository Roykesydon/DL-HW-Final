from domain.stock_dataset import StockDataset
from domain.stock_api import StockAPI
import datetime


class VirtualEnvironment:
    def __init__(self, train_end_date: str):
        self.train_end_date = train_end_date

        self.user_own_stock = []
        self.stock_time_index = {}
        self.complete_flag = {}
        self.property = 100000000

        self.stock_dataset = StockDataset(
            input_days=90, keep_columns=["open", "high", "low", "close"], output_days=1
        )

    def set_decision_module(self, decision_module):
        self.decision_module = decision_module

    def _lower_bound(self, search_list, target):
        left = 0
        right = len(search_list) - 1
        while left < right:
            mid = (left + right) // 2
            if search_list[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    def _buy_and_sell(self, stock_code, decision_list, data):
        # "open", "high", "low", "close"
        high = data[self.stock_time_index[stock_code]][1]
        low = data[self.stock_time_index[stock_code]][2]
        for decision in decision_list:
            if decision["price"] < low:
                # print(f"Decision price: {decision['price']}, low: {low}")
                continue
            if decision["price"] > high:
                # print(f"Decision price: {decision['price']}, high: {high}")
                continue

            intend = decision["intend"]
            if intend == "buy":
                single_fee = decision["price"] * 1000 * 1 * (1 + 0.001425)
                can_buy_shares = self.property // single_fee
                can_buy_shares = min(can_buy_shares, decision["shares"])

                self.property -= can_buy_shares * single_fee

                print(
                    f"Buy {can_buy_shares} shares of {stock_code} at {decision['price']}"
                )
                self.user_own_stock.append(
                    {
                        "stock_code_id": stock_code,
                        "beginning_price": decision["price"],
                        "shares": can_buy_shares,
                    }
                )
            elif intend == "sell":
                # check if own shares > decision["shares"]
                own_shares = 0
                for stock in self.user_own_stock:
                    if stock["stock_code_id"] == stock_code:
                        own_shares += stock["shares"]

                if own_shares < decision["shares"]:
                    continue

                # remove shares with lowest beginning price
                self.user_own_stock = sorted(
                    self.user_own_stock,
                    key=lambda x: x["beginning_price"],
                    reverse=False,
                )

                for stock in self.user_own_stock:
                    if decision["shares"] == 0:
                        break
                    if stock["stock_code_id"] == stock_code:
                        if decision["shares"] >= stock["shares"]:
                            money = (
                                decision["price"]
                                * stock["shares"]
                                * 1000
                                * (1 - 0.001425)
                            )
                            self.property += money
                            self.user_own_stock.remove(stock)
                            decision["shares"] -= stock["shares"]
                            print("Sell all shares")
                            print(
                                f"Sell {stock['shares']} shares of {stock_code} at {decision['price']}, beginning price: {stock['beginning_price']}"
                            )
                        else:
                            money = (
                                decision["price"]
                                * decision["shares"]
                                * 1000
                                * (1 - 0.001425)
                            )
                            self.property += money
                            print("sell some shares")

                            stock["shares"] -= decision["shares"]
                            print(
                                f"Sell {decision['shares']} shares of {stock_code} at {decision['price']}, beginning price: {stock['beginning_price']}"
                            )
                            decision["shares"] = 0
                            

    def run(self):
        all_data = self.stock_dataset.get_all_data()

        train_end_timestamp = datetime.datetime.strptime(
            self.train_end_date, "%Y%m%d"
        ).timestamp()

        lowest_result = 0
        highest_result = 0

        while True:
            if len(self.complete_flag) == len(all_data):
                break
            for stock_data in all_data:
                if stock_data["stock_code"] in self.complete_flag:
                    continue
                # print(f"Property: {self.property}")
                stock_code = stock_data["stock_code"]
                data = stock_data["data"]
                data_date = stock_data["data_date"]

                if stock_code not in self.stock_time_index:
                    train_end_index = self._lower_bound(data_date, train_end_timestamp)
                    if data_date[train_end_index] == train_end_timestamp:
                        train_end_index += 1

                    assert train_end_index < len(data)
                    self.stock_time_index[stock_code] = train_end_index
                if self.stock_time_index[stock_code] >= len(data):
                    # sell all stock
                    user_own_stock_with_same_code = [
                        x
                        for x in self.user_own_stock
                        if x["stock_code_id"] == stock_code
                    ]
                    for stock in user_own_stock_with_same_code:
                        highest_result += (
                            data[-1][1] * stock["shares"] * 1000 * (1 - 0.001425)
                        )
                        lowest_result += (
                            data[-1][2] * stock["shares"] * 1000 * (1 - 0.001425)
                        )

                    self.user_own_stock = [
                        x
                        for x in self.user_own_stock
                        if x["stock_code_id"] != stock_code
                    ]

                    self.complete_flag[stock_code] = True
                    break

                # get current user's stock with same stock code
                user_owned_stock_with_same_code = [
                    x for x in self.user_own_stock if x["stock_code_id"] == stock_code
                ]

                decision_data = data[: self.stock_time_index[stock_code]]

                decision_list = self.decision_module.make_decision(
                    stock_code, user_owned_stock_with_same_code, decision_data
                )

                # check decision list
                # if want to sell but not have stock, skip
                new_decision_list = []
                own_shares = 0
                cur_shares_count = 0
                
                for stock in user_owned_stock_with_same_code:
                    own_shares += stock["shares"]
                for decision in decision_list:
                    if decision["intend"] == "sell":
                        cur_shares_count += decision["shares"]
                    if cur_shares_count <= own_shares:
                        new_decision_list.append(decision)
                    else:
                        print(f"Skip Sell decision: {decision}")
                        
                decision_list = new_decision_list

                self._buy_and_sell(stock_code, decision_list, data)
                self.stock_time_index[stock_code] += 1
                
                assert self.property >= 0

        print(f"Property: {self.property}")
        print(f"Lowest result: {self.property+lowest_result}")
        print(f"Highest result: {self.property+highest_result}")
