from pydantic import BaseModel, Field
from typing import Optional


class Person(BaseModel):
    name:str= Field(description="name of the person")
    age:Optional[int] = Field(default= None, description="age of the person (1-100)")
    city:str = Field(description="city of the person (state)")

new_person = Person(name="Ali", age=23, city="alwer")

print(new_person.model_dump())

new_person1 = Person(name="Ali", city="alwer")

print(new_person1.model_dump())


# pydantic is smart enough that it will auto matcially type check and if you expected in int and but got in string it will automatcilly do type conversition.
# email checking is a build in feature in pydantic , if you pass invalid email it will raise an error.
