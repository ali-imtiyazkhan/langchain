from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int
    city:str

new_person : Person = {'name': "Ali", 'age': 23, 'city': 'alwer'}

print(new_person)
