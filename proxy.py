from domain.stock_api import StockAPI
from domain.stock_dataset import StockDataset
from domain.decision_module import DecisionModule
from domain.figure_generator import FigureGenerator
from configs.proxy_config import PROXY_CONFIG
from torch import tensor
import torch
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--skip",
    action="store_true",
    help="Skip the confirmation of buying and selling",
)
parser.set_defaults(skip=False)

args = parser.parse_args()

result_folder_name = PROXY_CONFIG["result_folder_name"]
TRAIN_END_DATE = PROXY_CONFIG["train_end_date"]

decision_module = DecisionModule()
stock_api = StockAPI()
stock_dataset = StockDataset(
    input_days=90, keep_columns=["open", "high", "low", "close"], output_days=1
)
figure_generator = FigureGenerator()

decision_module.load_model(f"./results/{result_folder_name}/weight.pth")
decision_module.set_mean_and_std(
    PROXY_CONFIG["mean_and_std"]
)

success, result = stock_api.get_user_stocks()
if not success:
    print("Failed to get user stocks")
    exit(1)

user_own_stock = result

def _buy_and_sell(stock_code, decision_list, data):
    # "open", "high", "low", "close"
    for decision in decision_list:
        intend = decision["intend"]
        if intend == "buy":
            fee = decision["price"] * 1000 * decision["shares"] * (1 + 0.001425)

            if fee > property:
                continue

            print("----")
            # generate figure
            print("You can see the recent 90 days data in ./tmp/proxy_recent_90_days.png")
            figure_generator.save_test_result_figure(
                f"./tmp/proxy_recent_90_days.png",
                {
                    "x": data[-90:],
                },
            )
            
            # check if user sure to buy
            print(f"Expected Sell Price: {decision['expected_price']}")
            if not args.skip:
                user_input = input(
                    f"Buy {decision['shares']} shares of {stock_code} at {decision['price']}? (y/N)"
                )
                if user_input != "y":
                    print("Skip buying")
                    continue

            fail_flag = False
            for _ in range(int(decision["shares"])):
                success = stock_api.buy(stock_code, 1, decision["price"])
                if not success:
                    print("Failed to buy")
                    fail_flag = True
                    break

            if fail_flag:
                continue

            print(
                f"Buy {decision['shares']} shares of {stock_code} at {decision['price']}"
            )

        elif intend == "sell":
            if decision["shares"] == 0:
                continue

            print("----")
            # generate figure
            print("You can see the recent 90 days data in ./tmp/proxy_recent_90_days.png")
            figure_generator.save_test_result_figure(
                f"./tmp/proxy_recent_90_days.png",
                {
                    "x": data[-90:],
                },
            )
            # check if user sure to sell
            if not args.skip:
                user_input = input(
                    f"Sell {decision['shares']} shares of {stock_code} at {decision['price']}, beginning price: {decision['beginning_price']}? (y/N)"
                )
                if user_input != "y":
                    print("Skip selling")
                    continue

            success = stock_api.sell(
                stock_code, decision["shares"], decision["price"]
            )
            
            if not success:
                print("Failed to sell")
                continue

            print(
                f"Sell {decision['shares']} shares of {stock_code} at {decision['price']}, beginning price: {decision['beginning_price']}"
            )


if __name__ == "__main__":
    all_data = stock_dataset.get_all_data()

    lowest_result = 0
    highest_result = 0

    # input user's property
    if not args.skip:
        property = int(input("Please input your property: "))
    else:
        property = 100000000

    for stock_data in all_data:
        stock_code = stock_data["stock_code"]
        data = stock_data["data"]
        data_date = stock_data["data_date"]

        # get current user's stock with same stock code
        user_owned_stock_with_same_code = [
            x for x in user_own_stock if x["stock_code_id"] == stock_code
        ]

        decision_data = data[-90:]

        decision_list = decision_module.make_decision(
            stock_code, user_owned_stock_with_same_code, decision_data
        )

        _buy_and_sell(stock_code, decision_list, data)

    # Evaluate asset
    for stock_data in all_data:
        stock_code = stock_data["stock_code"]
        data = stock_data["data"]
        data_date = stock_data["data_date"]

        user_own_stock_with_same_code = [
            x for x in user_own_stock if x["stock_code_id"] == stock_code
        ]
        for stock in user_own_stock_with_same_code:
            highest_result += data[-1][1] * stock["shares"] * 1000 * (1 - 0.001425)
            lowest_result += data[-1][2] * stock["shares"] * 1000 * (1 - 0.001425)

    print("----")
    print(f"Property: {property}")
    print(f"Lowest Evaluated Asset: {property+lowest_result}")
    print(f"Highest Evaluated Asset: {property+highest_result}")
