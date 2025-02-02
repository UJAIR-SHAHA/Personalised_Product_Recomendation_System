# Personalised_Product_Recomendation_System

## Overview
This project is a **Product Recommendation System** designed to provide users with personalized shopping experiences. It integrates **user-based** and **content-based recommendation techniques** along with a **deep learning model** for product recommendation, to suggest relevant products based on user interactions and preferences. The system is implemented as a **web application** using **Flask**.

---

## Features
1. **User Authentication**:
   - Users can log in and access their personalized recommendations.
2. **Content-Based Recommendations**:
   - Suggests products similar to the one currently being viewed.
3. **User-Based Recommendations**:
   - Recommends products based on the preferences of similar users.
4. **Product Search**:
   - Allows users to search for products by keywords.
5. **Interactive Dashboard**:
   - Displays trending products and user-specific suggestions.

---

## Technology Stack
- **Frontend**:
  - HTML/CSS for user interface design
  - JavaScript for interactive elements
- **Backend**:
  - Python and Flask for server-side logic
- **Data Processing**:
  - Pandas and NumPy for data manipulation
  - Scikit-learn for recommendation model development
- **Recommendation Techniques**:
  - TF-IDF for content similarity
  - TruncatedSVD for collaborative filtering
- **Database**:
  - SQL for storing user-product interactions
  - CSV files for product and user data

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/product-recommendation-system.git
   ```

2. Navigate to the project directory:
   ```bash
   cd product-recommendation-system
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   - Create a `.env` file in the project root and add the following:
     ```env
     FLASK_SECRET_KEY=your_secret_key
     ```

6. Run the Flask application:
   ```bash
   flask run
   ```

7. Open the application in your browser:
   - Navigate to `http://127.0.0.1:5000/`.

---

## File Structure
```
project-directory/
|-- static/
|   |-- img/          # Images for products and UI
|-- templates/
|   |-- *.html        # HTML templates for pages
|-- models/
|   |-- *.csv         # Product and user data files
|-- app.py            # Main application file
|-- requirements.txt  # Python dependencies
|-- .env              # Environment variables
```

---

## Usage
1. **Login**: Enter a valid user ID to access recommendations.
2. **Search Products**: Use the search bar to find products.
3. **View Similar Products**: Click on a product to see content-based recommendations.


---

## Future Enhancements
1. Expand product categories and data sources.
2. Include user feedback for improving recommendations.
3. Implement real-time data updates.

---

## Contributors
- Ujair Shaha ([GitHub Profile](https://github.com/ujair-shaha))

---

