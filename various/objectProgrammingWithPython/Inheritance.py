class Parent:
    def __init__(self, name):
        self.name = name
        print(f'Parent constructor, Name: {self.name}')

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
        print(f'Child constructor, Name: {self.name}')

child = Child('Zdzich', 4)
print(child.age)