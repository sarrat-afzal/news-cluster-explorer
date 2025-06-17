# app.py

from flask import Flask, render_template, request
from clustering import get_clusters_for_date
from datetime import date as _date

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """
    Main page: fetch & cluster news for the given date,
    then render the index.html template.
    """
    # 1) Get 'date' from query params or default to today
    date_str = request.args.get("date") or _date.today().isoformat()

    # 2) Run the clustering pipeline
    clusters = get_clusters_for_date(date_str, n_clusters=None)

    # 3) Render the template
    return render_template("index.html", clusters=clusters, date=date_str)

if __name__ == "__main__":
    # Run on localhost port 5001 with debug enabled
    app.run(debug=True, host="127.0.0.1", port=5001)


