from flask import Flask, render_template, request

# ADD IMPORTS HERE FOR ML LIBRARIES FOR IMPORTING MODELS AND PREDICTING STUFF
# eg. from scikit-learn import load_model
import pickle
import numpy as np
classifier =pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

TITLE = "Airline Passenger Satisfaction Prediction"

# IMPORT MODELS AND OTHER RELEVANT FILES HERE
# model = load_model("your_model.ext")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gender = int(request.form.get("gender"))
        customer_type = int(request.form.get("customer_type"))
        internet_facility = float(request.form.get("internet_facility"))
        overall_convenience_rating = float(request.form.get("overall_convenience_rating"))
        overall_service_quality_rating = float(request.form.get(
            "overall_service_quality_rating"
        ))
        overall_experience = float(request.form.get("overall_experience"))
        total_travel_class = int(request.form.get("total_travel_class"))
        age_group = int(request.form.get("age_group"))
        distance_category =int(request.form.get("distance_category"))
        total_delay_avg = float(request.form.get("total_delay_avg"))

        # USE THE DATA FROM ABOVE
        input=[gender,customer_type,internet_facility,overall_convenience_rating,
               overall_service_quality_rating,overall_experience,total_travel_class,
              age_group,distance_category,total_delay_avg]
        print(input)
        # DO PREDICTIONS HERE

        # PREDICT THIS
        satisfaction = classifier.predict([input])
        
        if satisfaction==1:
            satisfaction ="satisfied"
        else:
            satisfaction ="dissatisfied"
        return render_template(
            "index.html",
            title=TITLE,
            satisfaction=satisfaction,
        )

    return render_template("index.html", title=TITLE)


if __name__ == "__main__":
    app.run(debug=True)
