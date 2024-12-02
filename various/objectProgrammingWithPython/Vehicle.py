#zle importuje przy dziedziczeniu
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

