class Pokoj:
    def __init__(self, szerokosc, dlugosc):
        self.szerokosc=szerokosc
        self.dlugosc=dlugosc

    def policz_powierzchnie(self):
        return self.szerokosc * self.dlugosc

    def modify_dimensions(self, nowa_szerokosc=None, nowa_dlugosc=None):
        if nowa_szerokosc:
            self.nowa_szerokosc=nowa_szerokosc
        if nowa_dlugosc:
            self.nowa_dlugosc=nowa_dlugosc

moje=Pokoj(4, 'zolty')


moje.modify_dimensions(nowa_szerokosc='Zielony')

print(moje.nowa_szerokosc)