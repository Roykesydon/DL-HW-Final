import unittest

from domain.stock_api import StockAPI


class TestStockAPI(unittest.TestCase):
    def setUp(self):
        self.stock = StockAPI()

    def test_buy(self):
        for i in range(10):
            success = self.stock.buy("1213", 1, 7.0 + 0.1 * i)
        self.assertTrue(success)

    # def test_sell(self):
    #     success = self.stock.sell("1213", 1, 8)
    #     self.assertTrue(success)

    def test_get_info(self):
        success, data = self.stock.get_info("1213")
        self.assertTrue(success)
        self.assertTrue(len(data) > 0)

    def test_get_user_stocks(self):
        success, data = self.stock.get_user_stocks()
        print("user stocks")
        print(data)
        self.assertTrue(success)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
