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
import threading


# creating Flask App
app = Flask(__name__)
load_dotenv()
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_key_for_dev")

# Loading Data
product_data = pd.read_csv("models/final_fashion_data.csv")
trending_products = pd.read_csv("models/Top_rated_fashion.csv")
user_interaction = pd.read_csv("models/user_interaction_data.csv")
subset_user_interaction = user_interaction.sample(frac=0.05, random_state=42)


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_data['tags'])

with open('models/user_item_matrix.pickle', 'rb') as file:
    user_item_matrix = pickle.load(file)

model = tf.keras.models.load_model("models/collabarative_filtering_model.h5")

with open('models/user_mapping.json', 'rb') as file:
    user_mapping = json.load(file)

with open('models/product_mapping.json', 'rb') as file:
    product_mapping = json.load(file)

# List of Prices
prices = [200, 300, 100, 70, 90, 450, 800, 500]

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


# Function to search products based on a query
def search_products(data, query, top_n=12):
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


def recommend_content_based(product_id, data, tfidf_m, top_n=16):
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

    recommended_popular_products = product_data.iloc[[i for i in top_indices if i != 2]][
        ['product_id', 'product_name', 'Brand', 'masterCategory', 'imageUrl']
    ].drop_duplicates(subset='product_id')
    return recommended_popular_products


def user_based_recommendation(user_id, user_interaction_data, top_n=12):

    user_interaction_data['user_id'] = user_interaction_data['user_id'].astype(str)
    user_item_mat = user_interaction.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    user_similarity = cosine_similarity(user_item_mat)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_mat.index, columns=user_item_mat.index)
    similar_users = user_similarity_df.loc[str(user_id)].sort_values(ascending=False)[1:].head(top_n)
    # Get ratings of similar users
    similar_users_ratings = user_item_mat.loc[similar_users.index]
    # Predict ratings for the target user by averaging the ratings from similar users
    predicted_ratings = similar_users_ratings.T.dot(similar_users.values) / similar_users.values.sum()
    # Get the items the user has not rated yet
    user_rated_items = user_item_mat.loc[str(user_id)] > 0
    unrated_items = predicted_ratings[user_rated_items == False]
    # Sort unrated items by predicted rating
    recommended_items = unrated_items.sort_values(ascending=False).head(top_n)
    # Fetch product details for the recommended items
    recommended_product_details = product_data[product_data['product_id'].isin(recommended_items.index)][
        ['product_id', 'product_name', 'Brand', 'imageUrl']].drop_duplicates(subset='product_id')
    return recommended_product_details


# def model_based_recommendation(user_id, model, user_mapping, product_mapping, data, top_n=10):
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

def svd_recommendation(user_id, user_item_mat, user_map, product_map, data, top_n=12):
    """
    Recommend top-N items for a given user using SVD.
    """
    # Ensure the user ID exists in the user mapping
    if user_id not in user_map:
        raise ValueError(f"User ID {user_id} not found in user mapping.")

    # Map user and product IDs to their integer indices
    user_idx = user_map[user_id]
    product_indices = np.array(list(product_map.values()))

    # Convert interactions to a user-item matrix (efficiently)
    user_item_mat = user_item_mat.to_numpy()

    # Apply SVD (we only need the U and Vt matrices for the target user)
    U, sigma, Vt = svds(user_item_mat, k=5)  # Use k latent features
    sigma = np.diag(sigma)  # Convert sigma to a diagonal matrix

    # Directly compute predicted ratings for the target user (no need to compute for all users)
    user_latent_vector = U[user_idx, :]  # Get the specific latent vector for the user
    predicted_ratings = np.dot(user_latent_vector, np.dot(sigma, Vt))

    # Get the top-N recommendations using numpy's argsort for efficient sorting
    top_n_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    top_n_item_indices = product_indices[top_n_indices]

    # Map item indices back to product IDs (optimized using dictionary lookup)
    recommended_item_ids = [pid for idx in top_n_item_indices for pid, pid_idx in product_map.items() if pid_idx == idx]
    recommended_item_ids = [int(item_id) for item_id in recommended_item_ids]
    # Ensure the data['product_id'] column is of type int32 for efficient comparison
    data['product_id'] = data['product_id'].astype('int32')
    # Filter only recommended items that exist in product_data
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
    filtered_ratings = [predicted_ratings[product_map[pid]] if pid in product_map
                        else np.nan for pid in recommended_item_ids]

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
    recommended_products = None

    def timeout_handler():
        raise Exception("Operation timed out!")

    def handler(user_id, product_data):
        # Start a timer that will trigger the timeout_handler after 5 seconds
        timer = threading.Timer(10, timeout_handler)  # 5-second timeout
        timer.start()

        error_message = None

        try:
            # recommendation_products = svd_recommendation(user_id, user_item_matrix,
            # user_mapping, product_mapping, product_data)
            recommendation_products = product_data.sample(12,random_state=int(hash(str(user_id))) % 1000)

        except MemoryError as mem_err:
            print("Memory error: Out of memory!")
            error_message = "Memory error occurred during recommendation calculation."
            recommendation_products = product_data.head(12)

        except Exception as e:
            print("SVD recommendation took too long. Falling back to random products.")
            # Fallback to random products if an exception occurs
            error_message = "SVD recommendation took too long. Falling back to random products."
            recommendation_products = product_data.sample(12)
        finally:
            # Always cancel the timer when the operation finishes (either successfully or due to timeout)
            timer.cancel()

        return recommendation_products, error_message

    if user_id:
        try:
            recommended_products, error_message = handler(user_id, product_data)
        except Exception as e:
            # General exception handling for the entire block
            print(f"An error occurred in the indexredirect function: {e}")
            error_message = f"An unexpected error occurred: {e}"
            recommended_products = product_data.head(12)  # Fallback to random products
    else:
        error_message = "User not logged in."
        recommended_products = product_data.head(12)  # Fallback to random products

    # Generate trending items
    trending_items = recommend_popular_items(13)

    # Define random prices
    price = [400, 500, 600, 700, 1000, 1220, 1060, 5000, 3000, 4000]

    # Render the index template
    return render_template(
        'index.html',
        user_id=user_id,
        trending_products=trending_items,
        recommended_products=recommended_products,
        error_message=error_message,  # Pass the error message to the template
        truncate=truncate,
        random_price=random.choice(price)
    )


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
        search_query = request.form.get('search_query')
        content_based_rec = search_products(product_data, search_query, top_n=12)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)

        else:

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
