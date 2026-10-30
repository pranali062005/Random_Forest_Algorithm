import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained RandomForest model
try:
    with open('RandomForest.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# HTML template embedded directly so you don't need a templates folder
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Forest Prediction Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: #f8fafc;
            min-height: 100vh;
            padding: 40px 0;
        }
        .card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
        }
        .form-control {
            background-color: #0f172a;
            border: 1px solid #334155;
            color: #fff;
        }
        .form-control:focus {
            background-color: #0f172a;
            color: #fff;
            border-color: #6366f1;
        }
        label {
            color: #94a3b8;
            font-weight: 500;
        }
        .btn-custom {
            background: #6366f1;
            border: none;
            font-weight: 600;
        }
        .btn-custom:hover {
            background: #4f46e5;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="text-center mb-4">
                    <h1 class="fw-bold text-white">Workforce & Layoff Risk Predictor</h1>
                    <p class="text-muted">Enter company data to evaluate predictions using Random Forest</p>
                </div>

                <div class="card p-4 p-md-5">
                    <form action="/predict" method="POST">
                        <div class="row g-3">
                            <div class="col-md-4">
                                <label class="form-label">Company Name</label>
                                <input type="text" class="form-control" name="company_name" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Industry</label>
                                <input type="text" class="form-control" name="industry" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Country</label>
                                <input type="text" class="form-control" name="country" required>
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">Company Size</label>
                                <input type="text" class="form-control" name="company_size" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Month (1-12)</label>
                                <input type="number" class="form-control" name="month" value="1" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Year</label>
                                <input type="number" class="form-control" name="year" value="2026" required>
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">Layoffs Count</label>
                                <input type="number" step="any" class="form-control" name="layoffs_count" value="0">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Layoff Percentage</label>
                                <input type="number" step="any" class="form-control" name="layoff_percentage" value="0.0">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Open Roles</label>
                                <input type="number" step="any" class="form-control" name="open_roles" value="0">
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">Reason for Layoffs</label>
                                <input type="text" class="form-control" name="reason_for_layoffs">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">AI Automation Impact</label>
                                <input type="text" class="form-control" name="ai_automation_impact">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">AI Replacement Risk</label>
                                <input type="text" class="form-control" name="ai_replacement_risk">
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">Hiring Trend</label>
                                <input type="text" class="form-control" name="hiring_trend">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Remote Jobs Percentage</label>
                                <input type="number" step="any" class="form-control" name="remote_jobs_percentage" value="0.0">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Top Hiring Role</label>
                                <input type="text" class="form-control" name="top_hiring_role">
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">Stock Growth Percent</label>
                                <input type="number" step="any" class="form-control" name="stock_growth_percent" value="0.0">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Revenue Growth Percent</label>
                                <input type="number" step="any" class="form-control" name="revenue_growth_percent" value="0.0">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Salary Budget Change</label>
                                <input type="text" class="form-control" name="salary_budget_change">
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">AI Adoption Level</label>
                                <input type="text" class="form-control" name="ai_adoption_level">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Employee Sentiment</label>
                                <input type="text" class="form-control" name="employee_sentiment">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Job Security Score</label>
                                <input type="number" step="any" class="form-control" name="job_security_score" value="0.0">
                            </div>

                            <div class="col-12 mt-4">
                                <button type="submit" class="btn btn-custom btn-lg text-white w-100">Predict</button>
                            </div>
                        </div>
                    </form>

                    {% if prediction_text %}
                    <div class="alert alert-info mt-4 text-center fw-bold fs-5 bg-dark border-0 shadow" style="color: #818cf8 !important;">
                        {{ prediction_text }}
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        
        data = {
            'company_name': [form_data.get('company_name', '')],
            'industry': [form_data.get('industry', '')],
            'country': [form_data.get('country', '')],
            'company_size': [form_data.get('company_size', '')],
            'month': [float(form_data.get('month', 1))],
            'year': [float(form_data.get('year', 2026))],
            'layoffs_count': [float(form_data.get('layoffs_count', 0))],
            'layoff_percentage': [float(form_data.get('layoff_percentage', 0.0))],
            'reason_for_layoffs': [form_data.get('reason_for_layoffs', '')],
            'ai_automation_impact': [form_data.get('ai_automation_impact', '')],
            'ai_replacement_risk': [form_data.get('ai_replacement_risk', '')],
            'open_roles': [float(form_data.get('open_roles', 0))],
            'hiring_trend': [form_data.get('hiring_trend', '')],
            'remote_jobs_percentage': [float(form_data.get('remote_jobs_percentage', 0.0))],
            'top_hiring_role': [form_data.get('top_hiring_role', '')],
            'stock_growth_percent': [float(form_data.get('stock_growth_percent', 0.0))],
            'revenue_growth_percent': [float(form_data.get('revenue_growth_percent', 0.0))],
            'salary_budget_change': [form_data.get('salary_budget_change', '')],
            'ai_adoption_level': [form_data.get('ai_adoption_level', '')],
            'employee_sentiment': [form_data.get('employee_sentiment', '')],
            'job_security_score': [float(form_data.get('job_security_score', 0.0))]
        }
        
        df = pd.DataFrame(data)
        prediction = model.predict(df)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Prediction Result: {prediction}")
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)