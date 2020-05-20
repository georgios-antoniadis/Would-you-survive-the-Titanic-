# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:38:03 2020

@author: George
"""

import pandas as pd
import random
from csv import writer
#Retrieving the file and the data

my_file_url = 'C:/Users/George/Documents/Programming projects/Titanic/Other/Excel needed/Titanic (Responses) (4).xlsx'
data_1 = pd.read_excel(my_file_url)
data_in_list = data_1.values.tolist()
#Getting the length of the file to loop over it
length = len(data_in_list)
#Setting the variable for the random selection
genders = ['male','female']
numbers = [5,4,3,2,1,0]
fares = [525,250,190,140,40]
classes = ['First','Second','Third']
ports = ['Southampton', 'Cherbourg','Queenstown']
decks = ['A','B','C','D','F','G','unknown']
#Looping over them
for i in range(length):
    variable = data_in_list[i]
    #Getting the name of the customer
    name = variable[1]
    #Gender
    gender_1 = variable[2]
    if gender_1 == 'Άνδρας':
        gender = 'male'
    elif gender_1 == 'Γυναίκα':
        gender = 'female'
    else:
        gender = random.choice(genders)
    #Age NEEDS CHANGING
    age = variable[3]
    #Wealth
    money = variable[4]
    if money == '50 χιλιάδες και άνω':
        parch = 5
        fare = 525
        clas = 'First' 
        deck = 'A'
    elif money == '18 χιλιάδες και άνω':
        parch = 3
        fare = 250
        clas = 'Second'
        deck = 'B'
    elif money == '9 χιλιάδες και άνω':
        parch = 1
        fare = 190
        clas = 'Third'
        deck = 'F'
    elif money == 'Άνεργος/η':
        parch = 0
        fare = 140
        clas = 'unknown'
    elif money == 'Φοιτητής/ρια':
        parch = 0
        fare = 40
        clas = random.choice(classes)
        deck = random.choice(decks)
    elif money == 'Μαθητής/ρια':
        parch = random.choice(numbers)
        fare = random.choice(fares)
        clas = 'Second'
        deck = random.choice(decks)
    #Brothers and sisters
    number_of_siblings = variable[5]
    #Port
    #Alone
    al = variable[7]
    if al == 'Ναι ήμουν μόνος/η μου':
        alone = 'y'
    elif al == 'Όχι δεν ήμουν μόνος/η μου':
        alone = 'n'
    port = random.choice(ports)
    #Adding the names to a different file
    lst1 = name
    lst = [1,gender,age,number_of_siblings,parch,fare,clas,deck,port,alone]
    with open('my_customers.csv', 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
        csv_writer.writerow(lst)
    with open('my_customers_names.csv', 'a+', newline='') as write_obj1:
        csv_writer = writer(write_obj1)
            # Add contents of list as last row in the csv file
        csv_writer.writerow(lst1)
            #['survived','sex','age','n_siblings_spouses','parch','fare','class','deck','embark_town','alone'])
        #lst = ['1', gender, age, idk, 7.25, 100, wealth, deck, 'Southampton', 'n']