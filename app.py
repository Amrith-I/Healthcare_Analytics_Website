from flask import Flask, render_template, request, redirect, session
from activities import register_user, login_user, check_admin_exists
from dashboard_functionality import classify_disease, cost_analysis, prescription_analysis

app = Flask(__name__)
app.secret_key = 'secret-key'  # Secret key for session management

@app.route('/')
def index():
    # Check if the session has an active username
    if 'username' in session:
        role = session.get('role', None)
        if role == 'Admin':
            return redirect('/dashboard')  # Redirect Admin to the dashboard
        elif role == 'User':
            return redirect('/dashboard')  # Redirect User to the dashboard

    # If no session exists, render the homepage
    return render_template('index.html', username=None, role=None)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error_message = None
    admin_exists = check_admin_exists()  # Check if Admin exists
    if request.method == 'POST':
        result = register_user(
            username=request.form['username'],
            email=request.form['email'],
            role=request.form['role'],
            password=request.form['password'],
            confirm_password=request.form['confirm_password']
        )
        if result['status'] == 'success':
            return redirect('/login')
        error_message = result['message']
    return render_template('register.html', error_message=error_message, admin_exists=admin_exists)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'POST':
        result = login_user(
            username=request.form['username'],
            password=request.form['password']
        )
        if result['status'] == 'success':
            session['username'] = result['username']
            session['role'] = result['role']
            return redirect('/dashboard')
        error_message = result['message']
    return render_template('login.html', error_message=error_message)


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect('/login')

    role = session['role']
    if role == 'Admin':
        return render_template('admin_dashboard.html', username=session['username'])
    else:
        return render_template('user_dashboard.html', username=session['username'])


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# Dashboard functionality Begining
@app.route("/")
def home():
    return render_template("admin_dashboard.html")

@app.route("/disease_diagnosis")
def disease_diagnosis():
    report, data = classify_disease()  # Get the classification report and sample data
    return render_template(
        "disease_diagnosis.html",
        report=report,
        data=data.to_dict(orient="records"),
        image_url="/static/images/disease_distribution.png"
    )

@app.route("/cost_analysis")
def cost_analysis_page():
    report, data = cost_analysis()
    return render_template(
        "cost_analysis.html",
        report=report,
        data=data.to_dict(orient="records"),
        image_url="/static/images/cost_analysis.png"
    )

@app.route("/prescription_analysis")
def prescription_analysis_page():
    report, data = prescription_analysis()
    return render_template(
        "prescription_analysis.html",
        report=report,
        data=data.to_dict(orient="records"),
        image_url="/static/images/prescription_analysis.png"
    )

if __name__ == '__main__':
    app.run(debug=True)
