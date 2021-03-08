import random


class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.stock = []
        self.mutual_funds = []
        self.history = []

    def addCash(self, amount_added):
        self.cash += amount_added
        self.history.append(f" {amount_added} $ is added to portfolio")
        return self.cash

    def removeCash(self, amount_removed):
        self.cash = self.cash - amount_removed
        self.history.append(f" {amount_removed} $ is removed from portfolio")
        return self.cash

    def buyStock(self, share, symbol, price):
        global list_stock
        list_stock = [share, symbol, price]
        list_stock.append([share, symbol, price])
        self.stock.append([share, symbol])
        self.history.append(f" [{share}, {symbol}] is added to portfolio")
        self.cash = self.cash - (share * price)
        return self.cash

    def sellStock(self, share, symbol):
        for stock in self.stock:
            if symbol == stock[1] and stock[0] > share:
                stock[0] = stock[0] - share
                self.history.append(f" [{share}, {symbol}] is sold from portfolio")
            elif share == stock[0]:
                self.stock.remove(stock)
                self.history.append(f" [{share}, {symbol}] is sold from portfolio")
            else:
                None
        x = random.uniform(0.5 * list_stock[2], 1.5 * list_stock[2])
        self.cash += x * share
        return self.cash

    def buyMutualFund(self, share, symbol):
        self.mutual_funds.append([share, symbol])
        self.history.append(f" [{share}, {symbol}] is added to portfolio")
        self.cash = self.cash - (share * 1)
        return self.cash

    def sellMutualfund(self, share, symbol):
        for fund in self.mutual_funds:
            if symbol == fund[1] and fund[0] > share:
                fund[0] = fund[0] - share
                self.history.append(f" [{share}, {symbol}] is sold from portfolio")
            elif share == fund[0]:
                self.mutual_funds.remove(fund)
                self.history.append(f" [{share}, {symbol}] is sold from portfolio")
            else:
                None
        x = random.uniform(0.9, 1.2)
        self.cash += x * share
        return self.cash

    def portfolio_list(self):
        return {"Cash": self.cash, "Stock": self.stock, "Mutual Funds": self.mutual_funds}

    def history_list(self):
        return "\n".join(self.history)


x = Portfolio(300)

x.addCash(50)

x.buyStock(5, "HFH", 20)
x.buyMutualFund(10.3, "BRT")
x.buyMutualFund(2, "GHT")
x.sellMutualfund(3, "BRT")
x.sellStock(1, "HFH")
x.removeCash(50)

print(x.portfolio_list())
print(x.history_list())
