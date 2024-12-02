class General:
    general_age = 18
    num_voters = 0

    def __init__(self, age):
        self.age = age
        General.num_voters += 1

    @staticmethod
    def is_eligible(age):
        return age >= General.general_age

    @classmethod
    def get_total_voters(cls):
        return cls.num_voters

    def check_eligibility(self):
        if General.is_eligible(self.age):
            print(f"You are eligible to vote at age {self.age}.")
        else:
            print(f"Apologies, but You are not eligible to vote at age {self.age}.")

voter1 = General(14)
voter2 = General(20)

print(f"total voters: {General.num_voters}")
print(voter1.check_eligibility())
print(voter2.check_eligibility())