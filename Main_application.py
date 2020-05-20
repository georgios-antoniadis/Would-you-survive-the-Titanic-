# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:19:34 2019

@author: George
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import pandas as pd 
import tkinter as tk
import csv
import pyglet
from csv import writer

#Show button
def show_entry_fields():
    print(first_name, last_name, gender, age, n_of_siblings, wealth, deck)
    print(data)

#Save the passenger as csv
def save_passenger():
    global my_customer
    my_customer = 'new_customer1.csv'
    global my_customer_url
    my_customer_url = 'C:/Users/George/Documents/Programming projects/Titanic/new_customer.csv'
    with open('new_customer1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['survived','sex','age','n_siblings_spouses','parch','fare','class','deck','embark_town','alone'])
        writer.writerow(['1', gender, age, n_of_siblings, 7.25, 100, wealth, deck, 'Southampton', 'n'])   
        

#Create new passenger button
def new_passenger():
    global e1
    global e2
    global e3
    global e4
    global e5
    global e6
    global e7
    
    button = tk.Tk()
 
    tk.Label(button, text='First Name').grid(row=0) 
    tk.Label(button, text='Last Name').grid(row=1)
    tk.Label(button, text='Gender').grid(row=2) 
    tk.Label(button, text='Age').grid(row=3)
    tk.Label(button, text='Number of siblings').grid(row=4) 
    tk.Label(button, text='Class').grid(row=5)
    tk.Label(button, text='Deck').grid(row=6) 
    e1 = tk.Entry(button) 
    e2 = tk.Entry(button)
    e3 = tk.Entry(button) 
    e4 = tk.Entry(button) 
    e5 = tk.Entry(button)
    e6 = tk.Entry(button)
    e7 = tk.Entry(button) 
    e1.grid(row=0, column=1) 
    e2.grid(row=1, column=1)
    e3.grid(row=2, column=1) 
    e4.grid(row=3, column=1) 
    e5.grid(row=4, column=1) 
    e6.grid(row=5, column=1) 
    e7.grid(row=6, column=1)
    
    tk.Button(button, text='Show', command=show_entry_fields).grid(row=7, column=1, sticky=tk.W, pady=4)
                                                
    tk.Button(button, text='Quit', command=button.quit).grid(row=7, column=0, sticky=tk.W, pady=4)

    tk.Button(button, text='Get', command=get_val).grid(row=7, column=2, sticky=tk.W, pady=4)
    
    tk.Button(button, text='Save', command=save_passenger).grid(row=7, column=0, sticky=tk.W, pady=4)

    tk.mainloop() 
    
#Get the values button
def get_val():
    global first_name
    global last_name
    global gender
    global age
    global n_of_siblings
    global wealth
    global deck
    global data
    first_name = e1.get()
    last_name = e2.get()
    gender = e3.get()
    age = int(e4.get())
    n_of_siblings = int(e5.get())
    wealth = e6.get()
    deck = e7.get()
    data = [first_name, last_name, gender, age, n_of_siblings, wealth, deck]

def my_google_forms():
    global my_google_answers
    my_google_answers = model.predict(my_google_data)
    for i in range(len(my_google_answers)):
        my_list = my_google_answers[i]
        with open('my_google_out.csv', 'a+', newline='') as write_obj2:
            csv_writer = writer(write_obj2)
                # Add contents of list as last row in the csv file
            csv_writer.writerow(my_list)
        
    
def project():
    global my_prediction
    my_prediction = model.predict(my_data)
    if my_prediction[0] > 0.5:
        animation_s = pyglet.image.load_animation('survived.gif')

        animSprite_s = pyglet.sprite.Sprite(animation_s)

        w = animSprite_s.width
        h = animSprite_s.height
    
        s_window = pyglet.window.Window(width = w, height = h)
    
        r,g,b,alpha = 0.5, 0.5, 0.8, 0.5
    
        @s_window.event
        def on_draw():
            s_window.clear()
            animSprite_s.draw()

        pyglet.app.run()
        
    else:
        animation_d = pyglet.image.load_animation('death.gif')

        animSprite_d = pyglet.sprite.Sprite(animation_d)

        w = animSprite_d.width
        h = animSprite_d.height
    
        d_window = pyglet.window.Window(width = w, height = h)
    
        r,g,b,alpha = 0.5, 0.5, 0.8, 0.5
    
        @d_window.event
        def on_draw():
            d_window.clear()
            animSprite_d.draw()

        pyglet.app.run()
        

def new_win():
    img = pyglet.image.load('titanic.jpg')
    animSprite_b = pyglet.sprite.Sprite(img)
    r,g,b,alpha = 0.5, 0.5, 0.8, 0.5
    
    main_window = pyglet.window.Window(width = 1920, height = 1080, fullscreen = True)  
    
    @main_window.event
    def on_draw():
        main_window.clear()
        animSprite_b.draw()

    pyglet.app.run()    
        
#Get the prediction button

#MY DATA IS RIGHT HERE
my_customer = 'new_customer1.csv'
my_customer_url = 'C:/Users/George/Documents/Programming projects/Titanic/Other/Excel needed/new_customer1.csv'
mine_file_path = (my_customer, my_customer_url)
#Added on 23rd of february
my_google = 'my_customers.csv'
my_google_url = 'C:/Users/George/Documents/Programming projects/Titanic/Total/my_customers.csv'
my_google_path = (my_google, my_google_url)

#Getting the data from the online database
TRAIN_DATA_URL = 'C:/Users/George/Documents/Programming projects/Titanic/Other/Excel needed/training.csv'
TEST_DATA_URL = 'C:/Users/George/Documents/Programming projects/Titanic/Other/Excel needed/eval.csv'
#Setting the paths
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)
#Identidying the only column we need to define specifically as it is the output
    
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5, # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True, 
                **kwargs)
    return dataset

#The datasets
raw_mine_data = get_dataset(mine_file_path)   
raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
#Created from google forms
raw_my_google_data = get_dataset(my_google_path)

#Data processing
#Packing all of the numerical values together
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names
            
    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        
        return features, labels

#Picking the numeric features
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']
    
packed_train_data = raw_train_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))
    
packed_test_data = raw_test_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

packed_my_data = raw_mine_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

packed_my_google_data = raw_my_google_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

example_batch, labels_batch = next(iter(packed_train_data))

#Since the data is continuous, we have to normalize it
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
desc

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data-mean)/std

    # See what you just created.
    #Creating a numeric column containing all of that info
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column

example_batch['numeric']
    
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()

    #Data that is only allowed some certain values is called categorical
    #What are the categories and the options
CATEGORIES = {
        'sex': ['male', 'female'],
        'class' : ['First', 'Second', 'Third'],
        'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone' : ['y', 'n']
        }

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    #This is gonna be one of the input layers later on
categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])

    #It is finally time to create the multilayer tree
print("This is where the tree begins")
print("=================================")
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])

    #Building the prossesing model
model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
    
model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data
my_data = packed_my_data
my_google_data = packed_my_google_data
#epochs is the number of iterations
model.fit(train_data, epochs=20)

print("Model is trained")
print("==================================")

#Testing and printing the accuracy
test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)
    
# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))
        
        
#Main window loop
#%%    
master = tk.Tk()
#Initiating the main window
# set window size
master.geometry("1080x720")
#Loading the image
tk.Button(master, text = 'Add new passenger', command = new_passenger, height = 5, width = 50).place(relx = 0.35, rely = 0.1)
tk.Button(master, text = 'Make prediction', command = project, height = 5, width = 50).place(relx = 0.35, rely = 0.3)
tk.Button(master, text = 'Open', command = new_win, height = 5, width = 50).place(relx = 0.35, rely = 0.5)
tk.Button(master, text = 'Google', command = my_google_forms, height = 5, width = 50).place(relx = 0.35, rely = 0.7)

master.mainloop()         