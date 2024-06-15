import torch
from copy import deepcopy


class DecisionModule:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.INPUT_DAYS = 90
        self.OUTPUT_DAYS = 10
        self.ACCEPT_SELL_PROFIT_THRESHOLD = 10000
        self.ACCEPT_BUY_PROFIT_THRESHOLD = 30000
        self.SINGLE_STOCK_FUNDS = 10000000

    def set_mean_and_std(self, mean_and_std: dict):
        self.mean_and_std = mean_and_std

    def make_decision(
        self, stock_code, user_owned_stock_with_same_code, decision_data
    ) -> list:
        if self.model is None:
            raise Exception("Model has not been loaded to DecisionModule")

        """
        Predict
        """
        input_data = deepcopy(decision_data[-self.INPUT_DAYS :])
        input_data = input_data.unsqueeze(0).to(self.device)

        # preprocess data
        mean = self.mean_and_std[stock_code]["mean"]
        std = self.mean_and_std[stock_code]["std"]
        input_data = (input_data - mean) / std

        output = self.model(input_data)

        # denormalize
        output = output * std + mean
        input_data = input_data * std + mean
        
        last_predict = output[0][-1][-1].cpu().detach().numpy()
        input_last_data = input_data[0][-1].cpu().detach().numpy()
        # "open", "high", "low", "close"

        """
        Calculate information
        """
        last_predict_high = last_predict[1]
        last_predict_low = last_predict[2]
        
        input_last_high = input_last_data[1]
        input_last_low = input_last_data[2]

        own_shares = 0
        for stock in user_owned_stock_with_same_code:
            own_shares += stock["shares"]

        """
        make decision
        """
        decision_list = []

        # decision selling
        selling_price = input_last_low + (input_last_high - input_last_low) * 0.8
        selling_handling_fee = 0.001425 * selling_price * own_shares * 1000
        for stock in user_owned_stock_with_same_code:
            begin_price = stock["beginning_price"]
            begin_buying_handling_fee = 0.001425 * begin_price * stock["shares"] * 1000
            selling_profit = (
                (selling_price - begin_price) * stock["shares"] * 1000
                - selling_handling_fee
                - begin_buying_handling_fee
            )
            expected_selling_price = (
                self.ACCEPT_SELL_PROFIT_THRESHOLD
                + selling_handling_fee
                + begin_buying_handling_fee
            ) / (stock["shares"] * 1000) + begin_price
            if selling_profit > self.ACCEPT_SELL_PROFIT_THRESHOLD:
                decision_list.append(
                    {
                        "intend": "sell",
                        "price": selling_price,
                        "shares": stock["shares"],
                        "beginning_price": begin_price,
                    }
                )
            else:
                # check if the stock has been in loss for 3 days
                last_three_days = decision_data[-3:]
                if all([x[2] < begin_price for x in last_three_days]):
                    decision_list.append(
                        {
                            "intend": "sell",
                            "price": begin_price * 1.005,
                            "shares": stock["shares"],
                            "beginning_price": begin_price,
                        }
                    )
                    continue
                
                decision_list.append(
                    {
                        "intend": "sell",
                        "price": expected_selling_price,
                        "shares": stock["shares"],
                        "beginning_price": begin_price,
                    }
                )

        # decision buying
        # buying_price = first_low + (first_high - first_low) * 0.55
        buying_price = input_last_low + (input_last_high - input_last_low) * 0.55
        buying_shares = max((self.SINGLE_STOCK_FUNDS // (buying_price * 1000)) - own_shares, 0)

        expected_selling_price = last_predict_low + (last_predict_high - last_predict_low) * 0.5

        buy_handling_fee = 0.001425 * buying_price * buying_shares * 1000
        expected_selling_handling_fee = 0.001425 * last_predict_high * own_shares * 1000

        # print all shape
        expected_profit = (
            (expected_selling_price - buying_price) * buying_shares * 1000
            - buy_handling_fee
            - expected_selling_handling_fee
        )

        if expected_profit > self.ACCEPT_BUY_PROFIT_THRESHOLD:
            decision_list.append(
                {
                    "intend": "buy",
                    "price": buying_price,
                    "shares": buying_shares,
                    "expected_price": expected_selling_price,
                }
            )
        return decision_list

    def load_model(self, model_path):
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()
