import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model from model.pkl
MODEL_PATH = "RandomForest.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# HTML Template with an attractive, responsive Tailwind CSS layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Layoff & AI Impact Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between">
    <div class="container mx-auto px-4 py-8 max-w-4xl">
        <!-- Header -->
        <header class="text-center mb-10">
            <h1 class="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-500 mb-2">
                Corporate Risk & Layoff Predictor
            </h1>
            <p class="text-slate-400 text-sm">Powered by your Scikit-Learn Random Forest Classifier & Flask</p>
        </header>

        <!-- Main Card -->
        <div class="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-6 md:p-8">
            <form method="POST" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <!-- Numerical Inputs -->
                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">Layoffs Count</label>
                        <input type="number" name="layoffs_count" required value="150"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">Layoff Percentage (%)</label>
                        <input type="number" step="0.1" name="layoff_percentage" required value="15.0"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">AI Replacement Risk (0-1)</label>
                        <input type="number" step="0.01" name="ai_replacement_risk" required value="0.45"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">Remote Jobs Percentage (%)</label>
                        <input type="number" step="0.1" name="remote_jobs_percentage" required value="30.0"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">Stock Growth Percent (%)</label>
                        <input type="number" step="0.1" name="stock_growth_percent" required value="5.2"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">Revenue Growth Percent (%)</label>
                        <input type="number" step="0.1" name="revenue_growth_percent" required value="2.1"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">Job Security Score (0-100)</label>
                        <input type="number" step="0.1" name="job_security_score" required value="65.0"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-300 mb-1">Company Size (Encoded/Numeric)</label>
                        <input type="number" name="company_size" required value="2"
                            class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>

                </div>

                <!-- Hidden or placeholder fields for categorical attributes expected by pipeline/model features -->
                <!-- Note: Adjust these default categorical mapping fields depending on how your preprocessor handles categorical data -->
                <input type="hidden" name="company_name" value="0">
                <input type="hidden" name="industry" value="0">
                <input type="hidden" name="country" value="0">
                <input type="hidden" name="month" value="6">
                <input type="hidden" name="year" value="2026">
                <input type="hidden" name="reason_for_layoffs" value="0">
                <input type="hidden" name="ai_automation_impact" value="0">
                <input type="hidden" name="open_roles" value="5">
                <input type="hidden" name="hiring_trend" value="1">
                <input type="hidden" name="top_hiring_role" value="0">
                <input type="hidden" name="salary_budget_change" value="0">
                <input type="hidden" name="ai_adoption_level" value="1">
                <input type="hidden" name="employee_sentiment" value="3">

                <!-- Submit Button -->
                <div class="mt-8">
                    <button type="submit" 
                        class="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold rounded-lg shadow-lg transition duration-200 transform hover:-translate-y-0.5">
                        Run Model Prediction
                    </button>
                </div>
            </form>

            <!-- Results Section -->
            {% if prediction is not none %}
            <div class="mt-8 p-6 bg-slate-900 border {% if prediction == 1 %}border-red-500/50 bg-red-950/20{% else %}border-emerald-500/50 bg-emerald-950/20{% endif %} rounded-xl text-center">
                <h3 class="text-lg font-semibold text-slate-300 mb-1">Prediction Result</h3>
                <p class="text-3xl font-bold {% if prediction == 1 %}text-red-400{% else %}text-emerald-400{% endif %}">
                    {% if prediction == 1 %} High Risk / Positive Class Detected {% else %} Low Risk / Negative Class Detected {% endif %}
                </p>
                <p class="text-xs text-slate-400 mt-2">Raw Model Output Code: {{ prediction }}</p>
            </div>
            {% endif %}
        </div>
    </div>

    <!-- Footer -->
    <footer class="text-center py-6 text-slate-500 text-xs">
        Deployed on Render • Flask Web Application
    </footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Extract numerical parameters from form matching model's expected 21 features order
            features = [
                float(request.form.get("company_name", 0)),
                float(request.form.get("industry", 0)),
                float(request.form.get("country", 0)),
                float(request.form.get("company_size", 0)),
                float(request.form.get("month", 0)),
                float(request.form.get("year", 0)),
                float(request.form.get("layoffs_count", 0)),
                float(request.form.get("layoff_percentage", 0)),
                float(request.form.get("reason_for_layoffs", 0)),
                float(request.form.get("ai_automation_impact", 0)),
                float(request.form.get("ai_replacement_risk", 0)),
                float(request.form.get("open_roles", 0)),
                float(request.form.get("hiring_trend", 0)),
                float(request.form.get("remote_jobs_percentage", 0)),
                float(request.form.get("top_hiring_role", 0)),
                float(request.form.get("stock_growth_percent", 0)),
                float(request.form.get("revenue_growth_percent", 0)),
                float(request.form.get("salary_budget_change", 0)),
                float(request.form.get("ai_adoption_level", 0)),
                float(request.form.get("employee_sentiment", 0)),
                float(request.form.get("job_security_score", 0))
            ]
            
            if model:
                # Reshape for single sample prediction
                input_data = np.array([features])
                pred = model.predict(input_data)
                prediction = int(pred[0])
        except Exception as e:
            print(f"Prediction error: {e}")

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)