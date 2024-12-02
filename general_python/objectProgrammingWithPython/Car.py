class Vehicle:
    def __init__(self, color, weight, mileage):
        self.color = color
        self.weight = weight
        self.mileage = mileage
        print('Vehicle constructor')

    def ride(self, distance):
        self.mileage += distance

    def add_comment(self, comment):
        self.comments.append('numer: 1')

class Car(Vehicle):
    def __init__(self, color, weight, mileage, cost, fuel_type, wheels=4):
        super().__init__(color, weight, mileage)
        self.cost = cost
        self.fuel_type = fuel_type
        print('Car constructor')

car1 = Car("Zolty", 1000, 15000, 5000, '95')
car2 = Vehicle('Bialy', 12000, 1000)
print(car1.mileage)
print(car2.color)
