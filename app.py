from flask import Flask, request, render_template, url_for, redirect, session
import pandas as pd
import random
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import json
from scipy.sparse.linalg import svds

# creating Flask App
app = Flask(__name__)
load_dotenv()
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_key_for_dev")



# Loading Data
product_data = pd.read_csv("models/final_fashion_data.csv")
trending_products = pd.read_csv("models/Top_rated_fashion.csv")
user_interaction = pd.read_csv("models/user_interaction_data.csv")


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_data['tags'])

# Generate user-item matrix (assuming product_data contains user-product interactions)
# user_item_matrix = product_data.pivot_table(index='user_id', columns='ProdID', values='Rating', fill_value=0)

with open('models/user_item_matrix.pickle', 'rb') as file:
    user_item_matrix = pickle.load(file)

model = tf.keras.models.load_model("models/collabarative_filtering_model.h5")

with open('models/user_mapping.json', 'rb') as file:
    user_mapping = json.load(file)

with open('models/product_mapping.json', 'rb') as file:
    product_mapping = json.load(file)

# product_data.head()


# List of Prices
prices = [200, 300, 100, 70, 90, 450, 800, 500]


# function


def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


# Function to search products based on a query
def search_products(data, query, top_n=10):
    # Transform the query using the same TF-IDF vectorizer
    query_tfidf = tfidf_vectorizer.transform([query])

    # Compute cosine similarity between the query and product descriptions
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Get indices of the most similar products
    similar_product_indices = cosine_similarities.argsort()[:-top_n - 1:-1]

    # Get the top recommended products
    recommended_products = data.iloc[similar_product_indices]

    return recommended_products[['product_id', 'product_name', 'Brand', 'masterCategory', 'imageUrl']].drop_duplicates(
        subset='product_id')


def recommend_content_based(product_id, data, tfidf_m, top_n=15):
    # Get the index of the product
    product_id = int(product_id)
    idx = data[data['product_id'] == product_id].index[0]

    # Calculate cosine similarity for the target product with all others
    target_vector = tfidf_m[idx]
    sim_scores = cosine_similarity(target_vector, tfidf_matrix).flatten()

    # Get the indices of the top N most similar products
    top_indices = np.argpartition(-sim_scores, range(top_n + 1))[1:top_n + 1]

    # Fetch the product details for the recommended products
    recommended_products = data.iloc[top_indices][
        ['product_id', 'product_name', 'Brand', 'masterCategory', 'imageUrl']
    ].drop_duplicates(subset='product_id')

    return recommended_products


def recommend_popular_items(top_n):
    top_indices = trending_products.nlargest(top_n, "popularity_score").index

    recommended_products = product_data.iloc[top_indices][
        ['product_id', 'product_name', 'Brand', 'masterCategory', 'imageUrl']
    ].drop_duplicates(subset='product_id')

    print(type(recommended_products['product_id']))

    return recommended_products


# def user_based_recommendation(user_id, model, user_mapping, product_mapping, data, top_n=10):
#     """
#     Recommend top-N items for a given user based on the trained model.
#
#     Parameters:
#         user_id (str): The ID of the user for whom to recommend items.
#         model (Model): The trained recommendation model.
#         user_mapping (dict): A dictionary mapping user IDs to their integer indices.
#         top_n (int): Number of recommendations to generate (default: 10).
#     Returns:
#         pd.DataFrame: DataFrame containing the recommended products and their details.
#     """
#     # Check if the user ID exists in the mapping
#     if user_id not in user_mapping:
#         raise ValueError(f"User ID '{user_id}' not found in the mapping.")
#
#     # Get the mapped user index
#     user_idx = user_mapping[user_id]
#
#     # Prepare a list of all item indices
#     item_indices = np.array(list(product_mapping.values()))
#
#     # Create inputs for the model: replicate the user index for all items
#     user_input = np.full_like(item_indices, fill_value=user_idx, dtype=np.int32)
#     item_input = item_indices
#
#     # Predict ratings for all items
#     predicted_ratings = model.predict([user_input, item_input], verbose=0).flatten()
#
#     # Combine item indices and predicted ratings into a single array
#     recommendations = list(zip(item_indices, predicted_ratings))
#
#     # Sort recommendations by predicted rating in descending order
#     recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
#
#     # Get the original item IDs from the item mapping
#     recommended_item_ids = [list(product_mapping.keys())[list(product_mapping.values()).index(idx)] for idx, _ in
#                             recommendations]
#
#     print(recommended_item_ids)
#     recommended_item_ids = [int(item_id) for item_id in recommended_item_ids]
#
#     # Filter only recommended items that exist in product_data
#     recommended_products = data[data['product_id'].isin(recommended_item_ids)]
#
#     print(data['product_id'].dtype)
#
#     # If some products are missing in product_data, display a warning or handle it
#     if len(recommended_products) != top_n:
#         missing_items = set(recommended_item_ids) - set(recommended_products['product_id'])
#         print(type(missing_items))
#         print(f"Warning: Missing items in product data: {missing_items}")
#
#     # Add predicted ratings to the recommendations
#     recommended_products = recommended_products.copy()
#     recommended_products['PredictedRating'] = [rating for _, rating in recommendations]
#
#     return recommended_products[['product_id', 'product_name', 'imageUrl', 'Brand']]



# def user_based_recommendation(user_id, model, user_mapping, product_mapping, data, top_n=10):
#     """
#     Recommend top-N items for a given user based on the trained model.
#
#     Parameters:
#         user_id (str): The ID of the user for whom to recommend items.
#         model (Model): The trained recommendation model.
#         user_mapping (dict): A dictionary mapping user IDs to their integer indices.
#         product_mapping (dict): A dictionary mapping product IDs to integer indices.
#         data (pd.DataFrame): The product data, including product details.
#         top_n (int): Number of recommendations to generate (default: 10).
#     Returns:
#         pd.DataFrame: DataFrame containing the recommended products and their details.
#     """
#     # Ensure the user ID exists in the user mapping
#     if user_id not in user_mapping:
#         raise ValueError(f"User ID {user_id} not found in user mapping.")
#
#     # Get the mapped user index
#     user_idx = user_mapping[user_id]
#
#     # Prepare a list of all item indices
#     item_indices = np.array(list(product_mapping.values()))
#
#     # Create inputs for the model: replicate the user index for all items
#     user_input = np.full_like(item_indices, fill_value=user_idx, dtype=np.int32)
#     item_input = item_indices
#
#     # Predict ratings for all items (vectorized operation)
#     predicted_ratings = model.predict([user_input, item_input], verbose=0).flatten()
#
#     # Get top-N recommendations (vectorized sorting)
#     top_n_indices = np.argsort(predicted_ratings)[-top_n:][::-1]  # Get indices of top-N predictions
#     top_n_ratings = predicted_ratings[top_n_indices]
#     top_n_item_indices = item_indices[top_n_indices]
#
#     # Map item indices back to product IDs using product_mapping
#     recommended_item_ids = [list(product_mapping.keys())[list(product_mapping.values()).index(idx)] for idx in top_n_item_indices]
#     recommended_item_ids = [int(item_id) for item_id in recommended_item_ids]
#     # Filter only recommended items that exist in product_data (using efficient pandas filtering)
#     recommended_products = data[data['product_id'].isin(recommended_item_ids)].copy()
#
#     # print("Recommended Item IDs:", recommended_item_ids)
#     # print("Product IDs in recommended_products:", recommended_products['product_id'].tolist())
#     # print("Columns in recommended_products:", recommended_products.columns)
#     # print(recommended_products.head())
#
#     print(type(recommended_item_ids[0]))
#     print(data['product_id'].dtype)
#
#     # Handle missing items in the product data
#     if len(recommended_products) != top_n:
#         missing_items = set(recommended_item_ids) - set(recommended_products['product_id'])
#         print(type(missing_items))
#         print(f"Warning: Missing items in product data: {missing_items}")
#
#     # Add predicted ratings to the recommendations
#     recommended_products['PredictedRating'] = top_n_ratings
#
#     recommended_products.head()
#
#     # Return a DataFrame with relevant product details
#     return recommended_products[['product_id', 'product_name', 'imageUrl', 'Brand', 'PredictedRating']]

def svd_recommendation(user_id, user_item_matrix, user_mapping, product_mapping, data, top_n=10):
    """
    Recommend top-N items for a given user using SVD.
    """
    # Ensure the user ID exists in the user mapping
    if user_id not in user_mapping:
        raise ValueError(f"User ID {user_id} not found in user mapping.")

    # Map user and product IDs to their integer indices
    user_idx = user_mapping[user_id]
    product_indices = np.array(list(product_mapping.values()))

    # Convert interactions to a user-item matrix (efficiently)
    user_item_matrix = user_item_matrix.to_numpy()

    # Apply SVD (we only need the U and Vt matrices for the target user)
    U, sigma, Vt = svds(user_item_matrix, k=50)  # Use k latent features
    sigma = np.diag(sigma)  # Convert sigma to a diagonal matrix

    # Directly compute predicted ratings for the target user (no need to compute for all users)
    user_latent_vector = U[user_idx, :]  # Get the specific latent vector for the user
    predicted_ratings = np.dot(user_latent_vector, np.dot(sigma, Vt))

    # Get the top-N recommendations using numpy's argsort for efficient sorting
    top_n_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    top_n_item_indices = product_indices[top_n_indices]

    # Map item indices back to product IDs (optimized using dictionary lookup)
    recommended_item_ids = [pid for idx in top_n_item_indices for pid, pid_idx in product_mapping.items() if pid_idx == idx]
    recommended_item_ids = [int(item_id) for item_id in recommended_item_ids]

    # Ensure the data['product_id'] column is of type int32 for efficient comparison
    data['product_id'] = data['product_id'].astype('int32')

    print(data['product_id'].dtypes)
    print(type(recommended_item_ids))


    # Filter only recommended items that exist in product_data (optimized with `isin`)
    recommended_products = data[data['product_id'].isin(recommended_item_ids)].copy()

    # Check if any products were found
    if recommended_products.empty:
        print("No matching products found in the dataset!")
        return pd.DataFrame()  # Return empty DataFrame if no products found

    # Handle missing items in the product data
    missing_items = set(recommended_item_ids) - set(recommended_products['product_id'])
    if missing_items:
        print(f"Warning: Missing items in product data: {missing_items}")

    # Generate predicted ratings for the recommended products efficiently
    # No need for an explicit loop, use `map` for vectorized operation
    filtered_ratings = [predicted_ratings[product_mapping[pid]] if pid in product_mapping else np.nan for pid in recommended_item_ids]

    # Add predicted ratings to the recommendations
    recommended_products['PredictedRating'] = filtered_ratings

    # Return a DataFrame with relevant product details
    return recommended_products[['product_id', 'product_name', 'imageUrl', 'Brand', 'PredictedRating']]

@app.before_request
def make_session_permanent():
    session.permanent = False
# Flask route for the index_page (First Page)
@app.route("/")
@app.route('/index')
def indexredirect():
    user_id = session.get('user_id')
    if user_id:
        # Generate recommendations for the logged-in user
        recommended_products = svd_recommendation(user_id, user_item_matrix, user_mapping, product_mapping, product_data)
    else:
        # Default case if no user is logged in
        recommended_products = product_data.head(12)  # Fallback: Random products

    trending_items = recommend_popular_items(9)

    price = [400, 500, 600, 700, 1000, 1220, 1060, 5000, 3000, 4000]
    return render_template('index.html', user_id=user_id, trending_products=trending_items,
                           recommended_products=recommended_products.head(12), truncate=truncate,
                           random_price=random.choice(price))


# Flask route for the login_page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id")  # Get user ID from the form
        password = request.form.get("password")
        # password = str(password)  # Get user password from the form
        if user_id in user_mapping and password == user_id:
            session["user_id"] = user_id
            # Redirect to index with user ID as a query parameter or session
            return redirect(url_for("indexredirect", user_id=user_id))
        else:
            error_message = "Please enter a valid user ID"
            return render_template("login.html", error_message=error_message)
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    # Perform any cleanup if necessary (e.g., clearing session data)
    return redirect(url_for("indexredirect"))


@app.route('/main')
def main():
    return render_template('main.html')


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    user_id = session.get("user_id")

    if request.method == 'POST':
        prod = request.form.get('prod')
        content_based_rec = search_products(product_data, prod, top_n=10)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            # Create a list of random image URLs for each recommended product
            # random_product_image_urls_1 = [
            #     brand_image_map.get(product['Brand'], "static/img/img_1.png")
            #     # Use a default image if brand is not in the map
            #     for _, product in content_based_rec.iterrows()
            # ]
            print(content_based_rec)
            # print(random_product_image_urls_1)

            price = [400, 500, 600, 700, 1000, 1220, 1060, 5000, 3000, 4000]
            return render_template('main.html', user_id=user_id, content_based_rec=content_based_rec, truncate=truncate,

                                   random_price=random.choice(price))


@app.route("/View_Similar_Product")
def view_similar_product():
    user_id = session.get("user_id")

    product_id = request.args.get('product_id')
    if not product_id:
        # If product_id is missing, redirect or show an error message
        message = "Product ID is missing. Please try again."
        return render_template('main.html', message=message)

        # Call the recommendation function (ensure it handles invalid IDs)
    similar_recommendations = recommend_content_based(product_id, product_data, tfidf_matrix)

    if similar_recommendations.empty:
        # Handle the case where there are no recommendations
        message = "No recommendations available for this product."
        return render_template('main.html', message=message)

        # Process the recommendations
    # random_product_image_urls_1 = [
    #     brand_image_map.get(product['Brand'], "static/img/img_1.png")
    #     for _, product in similar_recommendations.iterrows()
    # ]

    # Random prices (or replace with dynamic data)
    price = [400, 500, 600, 700, 1000, 1220, 1060, 5000, 3000, 4000]

    # Return the rendered template with data
    return render_template(
        'similar_product.html', user_id=user_id,
        recommended_products=similar_recommendations,
        truncate=truncate,

        random_price=random.choice(price)
    )


# Run the Flask app
if __name__ == "__main__":
    app.run()
