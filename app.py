from flask import Flask, render_template, request
from main import process_input  # import your function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            result = process_input(user_input)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)