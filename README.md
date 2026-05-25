# 🎬 Movie Recommendation System

A modern Content-Based Movie Recommendation System built using Python, Machine Learning, and Streamlit that recommends similar movies based on genres, cast, keywords, director, and movie overview.

---

# 📌 Problem Statement

Finding movies similar to a user's interests from thousands of available movies can be difficult and time-consuming. Traditional search methods do not provide personalized content recommendations.

This project solves the problem by building a content-based recommendation engine that suggests movies similar to a selected movie using Machine Learning techniques.

---

# 🎯 Objective

The objective of this project is to build an intelligent movie recommendation system that:

* Recommends movies based on content similarity
* Improves movie discovery experience
* Demonstrates Machine Learning and NLP concepts
* Provides a fast and interactive web application using Streamlit

---

# ✨ Features

✅ Content-Based Movie Recommendation

✅ Interactive Streamlit Web Application

✅ Movie Posters using TMDb API

✅ Fast Recommendations using Precomputed Similarity Matrix

✅ Dropdown-Based Movie Selection

✅ Movie Details Preview

✅ Responsive Dark-Themed UI

✅ Optimized ML Architecture using Pickle Files

---

# 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* Requests
* Pickle
* TMDb API

---

# 🧠 Concepts Used

* Content-Based Filtering
* Natural Language Processing (NLP)
* Count Vectorizer
* Cosine Similarity
* Feature Engineering
* Machine Learning Pipeline
* Recommendation Systems

---

# ⚙️ Project Architecture

1. Movie metadata is collected and preprocessed
2. Important features are combined into tags
3. Count Vectorizer converts text data into vectors
4. Cosine Similarity calculates similarity scores
5. Similarity matrix is stored using Pickle
6. Streamlit application loads precomputed files for fast recommendations

---

# 📷 Project Screenshots

## 🏠 Home Page

![Home](screenshots/home.png)

## 🎬 Selected Movie Details

![Selected Movie](screenshots/selected_movie.png)

## ✨ Recommended Movies

![Recommendations](screenshots/recommendations.png)

---

# 📁 Project Structure

```bash
Movie-Recommendation-System/
│
├── app.py
├── movie_dict.pkl
├── similarity.pkl
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── secrets.toml
│
├── screenshots/
│   └── recommendations.png
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/vamsiandavarapu/Movie-Recommendation-System.git
```

Move into the project directory:

```bash
cd Movie-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run the Streamlit application using:

```bash
streamlit run app.py
```

After running, open the local URL displayed in the terminal.

---

# 🌐 Deployment

Deployed using Streamlit Community Cloud.

---

# 🔮 Future Improvements

* Add Hybrid Recommendation System
* Add Collaborative Filtering
* Add Movie Search Autocomplete
* Add Trailer Integration
* Add User Authentication
* Add Watchlist Feature

---

# 👨‍💻 Author

## Vamsi Andavarapu

* GitHub: https://github.com/vamsiandavarapu
* LinkedIn: https://www.linkedin.com/in/vamsiandavarapu
* Email: vamsiandavarapu83096@gmail.com

---

# ⭐ Support

If you found this project useful:

* Give this repository a ⭐ on GitHub
* Share your feedback
* Connect with me on LinkedIn

Your support is appreciated!
