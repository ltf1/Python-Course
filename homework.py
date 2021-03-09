import random


class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.stock = {}
        self.mutual_funds = {}
        self.history = []

    def addCash(self, amount_added):
        self.cash += amount_added
        self.history.append(f" {amount_added} $ is added to portfolio")
        return self.cash

    def withdrawCash(self, amount_withdraw):
        self.cash = self.cash - amount_withdraw
        self.history.append(f" {amount_withdraw} $ is withdrawn from portfolio")
        return self.cash

    def Stock(self, price, symbol):
        s = [price, symbol]
        return s

    def buyStock(self, share, s):
        if s[1] in self.stock:
            self.stock[s[1]] += share
        else:
            self.stock[s[1]] = share
        self.history.append(f" [{share}, {s[1]}] is added to portfolio")
        self.cash = self.cash - (share * s[0])
        return self.cash

    def sellStock(self, symbol, share):
        if symbol in self.stock and self.stock[symbol] > share:
            self.stock[symbol] -= share
            self.history.append(f" [{share}, {symbol}] is sold from portfolio")
        elif self.stock[symbol] == share:
            del self.stock[symbol]
            self.history.append(f" [{share}, {symbol}] is sold from portfolio")
        else:
            None
        random_num = random.uniform(0.5, 1.5) * s[0]
        self.cash = self.cash + random_num * share
        return self.cash

    def MutualFund(self, symbol):
        mf = symbol
        return mf

    def buyMutualFund(self, mf, share):
        if mf in self.mutual_funds:
            self.mutual_funds[mf] = round(self.mutual_funds[mf] + share, 2)
        else:
            self.mutual_funds[mf] = share
        self.history.append(f" [{share}, {mf}] is added to portfolio")
        self.cash = self.cash - (share * 1)
        return self.cash

    def sellMutualfund(self, symbol, share):
        if symbol in self.mutual_funds and self.mutual_funds[symbol] > share:
            self.mutual_funds[symbol] = round(self.mutual_funds[symbol] - share, 2)
            self.history.append(f" [{share}, {symbol}] is sold from portfolio")
        elif self.mutual_funds[symbol] == share:
            del self.mutual_funds[symbol]
            self.history.append(f" [{share}, {symbol}] is sold from portfolio")
        else:
            None
        random_num = random.uniform(0.9, 1.2)
        self.cash += random_num * share
        return self.cash

    def portfolio_list(self):
        return {"Cash": round(self.cash, 2), "Stock": self.stock, "Mutual Funds": self.mutual_funds}

    def history_list(self):
        return "\n".join(self.history)


portfolio = Portfolio(300)
s = portfolio.Stock(20, "GBT")
s2 = portfolio.Stock(15, "HBY")
mf1 = portfolio.MutualFund("HGB")
mf2 = portfolio.MutualFund("MLT")

portfolio.addCash(50)
portfolio.withdrawCash(20)
portfolio.addCash(400)

portfolio.buyStock(6, s)
portfolio.buyStock(3, s)
portfolio.buyStock(2, s2)
portfolio.buyStock(4, s2)
portfolio.sellStock("HBY", 2)
portfolio.sellStock("GBT", 2)

portfolio.buyMutualFund("HGB", 3.2)
portfolio.buyMutualFund("HGB", 3.2)
portfolio.sellMutualfund("HGB", 2.4)
portfolio.buyMutualFund("MLT", 8.9)
portfolio.sellMutualfund("MLT", 1.3)



print(portfolio.portfolio_list())
print(portfolio.history_list())
