# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:01:23 2024

@author: Admin
"""



##############################################
import numpy as np
#install Flask: pip install Flask

"""
Flask is a backend web framework based on the 
Python programming language. It basically allows 
the creation of web applications in a Pythonic 
syntax and concepts.
"""
"""
Flask supports generating dynamic HTML content
 via templates using the render_template() function.
It uses the Jinja2 template engine to do this.
"""

"""
Using Flask we can set up a web server to load up
some basic HTML templates along with Jinja2 templating syntax.
 """

from flask import Flask, request, render_template
"""
This Flask class instance will become our WSGI 
(Web Server Gateway Interface) application. 
WSGI is a specification that describes how a 
web server communicates with web applications.
"""

import pickle



#create an app object using Flask class
app = Flask(__name__)

#load the trained model,i.e. pickle file
model = pickle.load(open('models/reg_model.pkl','rb'))
"""
Create a route that receives html inputs,
uses the trained model to make a prediction, 
and returns that prediction,
which can be accessed through the API endpoint
"""
"""
The render_template() function makes it easy to 
serve dynamic HTML content with Flask. 
It can also accept variables, allowing you to 
customize the HTML content it serves.
"""
"""
The request object from flask HELPS to get the HTML form data.

"""
@app.route('/')
def home():
    return render_template('index.html')


"""
POST is a HTTP request method in web development to retrieve or request data from html form to backend
There is no limit on how much data you can send. 
POST also supports the sending of any kind of data.
"""
   
    
@app.route('/predict/',methods=['POST']) 
def predict():
    int_features=[float(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='Predicted salary is {}'.format(output))


# run the Flask application
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


