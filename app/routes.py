from flask import Blueprint, render_template, request, redirect, url_for, session
import os
import pandas as pd
from app.segmentation import get_segmentation_results
from app.recommendation import load_recommendation_data, recommend_items

main = Blueprint("main", __name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/choose_analysis", methods=["GET", "POST"])
def choose_analysis():
    if request.method == "POST":
        analysis_type = request.form["analysis_type"]
        session["analysis_type"] = analysis_type  # Store choice in session
        if analysis_type == "segmentation":
            return redirect(url_for("main.upload_segmentation"))
        return redirect(url_for("main.upload_recommendation"))
    return render_template("choose_analysis.html")

# ---------------- SEGMENTATION ----------------
@main.route("/upload_segmentation", methods=["GET", "POST"])
def upload_segmentation():
    required_fields = ['visitorid', 'total_views', 'total_addtocart', 'total_purchases']

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, "segmentation_data.csv")
            file.save(filepath)
            return redirect(url_for("main.process_segmentation"))

    return render_template("upload.html", analysis_type="segmentation", required_fields=required_fields)


@main.route('/process_segmentation', methods=['POST'])
def process_segmentation():
    file = request.files.get('file')
    if not file:
        return "Error: No file uploaded!", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    required_fields = ['visitorid', 'total_views', 'total_addtocart', 'total_purchases']

    df = pd.read_csv(filepath)
    if not all(field in df.columns for field in required_fields):
        return "Error: CSV file is missing required fields!", 400

    return redirect(url_for("main.segmentation_results"))

@main.route('/segmentation_results')
def segmentation_results():
    cluster_counts, recommendations, plot_data = get_segmentation_results()
    return render_template(
        "segmentation_results.html",
        cluster_counts=cluster_counts,
        recommendations=recommendations,
        plot_data=plot_data
    )

# ---------------- RECOMMENDATION ----------------
@main.route("/upload_recommendation", methods=["GET", "POST"])
def upload_recommendation():
    required_fields = ['visitorid', 'itemid', 'event', 'user_index', 'item_index', 'event_weight']

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, "recommendation_data.csv")
            file.save(filepath)
            return redirect(url_for("main.process_recommendation"))

    return render_template("upload.html", analysis_type="recommendation", required_fields=required_fields)


@main.route('/process_recommendation', methods=['POST'])
def process_recommendation():
    file = request.files.get('file')
    if not file:
        return "Error: No file uploaded!", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    required_fields = ['visitorid', 'itemid', 'event']

    df = pd.read_csv(filepath)
    if not all(field in df.columns for field in required_fields):
        return "Error: CSV file is missing required fields!", 400

    load_recommendation_data(filepath)
    return redirect(url_for("main.recommendation_input"))

@main.route('/recommendation', methods=['GET', 'POST'])
def recommendation_input():
    """User input page for recommendation."""
    if request.method == 'POST':
        visitorid = int(request.form.get('visitorid'))
        recommendations = recommend_items(visitorid)
        return render_template('recommendation_results.html', recommendations=recommendations)
    return render_template('recommendation_input.html')

@main.route('/recommendation_results', methods=['POST'])
def recommendation_results():
    visitorid = int(request.form.get('visitorid'))
    recommendations = recommend_items(visitorid)
    return render_template('recommendation_results.html', recommendations=recommendations)
