# 🧠 Depresso Predictor

**Depresso Predictor** is a Django-based web application that uses machine learning to predict possible signs of depression among college students based on survey data. It serves as an awareness tool, especially for youth mental health support and early self-assessment.

---

## 📁 Project Structure

```
.
├── espresso/           # Django project settings
├── espressoapp/        # Core app with ML logic and frontend
├── README.md           
├── db.sqlite3          # SQLite database
└── manage.py           # Django management script
├── path.code-workspace
```

---

## 🚀 Getting Started

Follow these steps to set up and run the app locally.

### 🔗 Clone the Repository

```bash
git clone https://github.com/hustinaa/DepressoPredictor.git
cd DeressoPredictor
```

### 🛠️ Set Up the Environment

1. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> Generate `requirements.txt` if not present:
> ```bash
> pip freeze > requirements.txt
> ```

---

### 🗄️ Run the Server

Apply migrations and start the server:

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Access the app via `http://127.0.0.1:8000`.

---

## 🧪 Features

- Depression prediction using Decision Tree and K-Nearest Neighbors (KNN)
- Input fields for survey-based factors (e.g., sleep, stress, lifestyle)
- Coffee & sunrise themed UI with yellow and orange color scheme
- Lightweight and intuitive results display for awareness

---

## ⚙️ Technologies Used

- **Backend**: Django (Python)
- **Frontend**: HTML, Tailwind CSS
- **Machine Learning**: scikit-learn (Decision Tree, KNN)
- **Database**: SQLite

---

## 👩‍💻 About the Developers

Developed by Justine Evora, Ronnel Bermas, Christine Sotoza. 
Currently studying **Management Information Systems** at **Ateneo de Manila University**.

If you find this project helpful, feel free to give it a ⭐ and connect!
