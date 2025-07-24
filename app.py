from flask import Flask, render_template, request
import pickle
import numpy as np

# 1. Create a Flask "app" which will handle web requests
app = Flask(__name__)

# 2. Load our trained model from disk once, when the server starts
with open('savedmodel.sav', 'rb') as f:
    model = pickle.load(f)

# 3. Map numerical predictions to human‑readable species names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# 4. Define a route for the home page ('/') that accepts GET and POST requests
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/', methods=['GET', 'POST']) # this means “when I go to / in my browser, run this function.”

# GET vs POST:
# A GET request just retrieves the page (shows the form).
# A POST request happens when you submit the form—your numbers go to the server.

def index():
    result = None
    image_filename = None

    if request.method == 'POST': # form was submitted 

        print("Form values:", request.form)

        try:
            # a) Read and convert form inputs 
            features = [
                float(request.form['sepal_length']), # request.form[...] reads what you typed ,, We convert strings to floats because flower measurements can be decimals.
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]

            print("Converted input:", features)

            sample = np.array([features])

            # b) Use model to predict
            outputs = model.predict(sample)
            pred_idx = int(np.argmax(outputs, axis=1)[0]) # pick the index of the highest probability.

            print("Pred:", pred_idx)

            # c) Map to species name
            result = species_map.get(pred_idx, 'Unknown')

            image_filename = None
            if result in species_map.values():
                image_filename = f"images/iris_{result}.jpg"

        except Exception:
            result = 'error' # on invalid input

    # 5. Render the HTML template to show the result, passing in our prediction
    return render_template('index.html', result=result, image_filename=image_filename)

if __name__ == '__main__':
    # 6. Start the development server on port 8080 --> makes your computer listen for web requests on localhost:8080.
    app.run(host='0.0.0.0', port=8080, debug=True)