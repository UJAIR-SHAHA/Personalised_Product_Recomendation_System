from flask import Flask, request, render_template, url_for, redirect, session
import pandas as pd
import random
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# creating Flask App
app = Flask(__name__)
load_dotenv()
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_key_for_dev")

# Loading Data
product_data = pd.read_csv("models/final_walmart_product_data.csv")
trending_products = pd.read_csv("models/Top_Rated.csv")

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_data['Tags'])

# Generate user-item matrix (assuming product_data contains user-product interactions)
user_item_matrix = product_data.pivot_table(index='user_id', columns='ProdID', values='Rating', fill_value=0)

# # Train the SVD model
# # Step 3: Apply SVD for Collaborative Filtering
# svd = TruncatedSVD(n_components=50)
# user_factors = svd.fit_transform(user_item_matrix)
# item_factors = svd.components_.T

# product_data.head()

# List of predefined images
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/covergirl.png",
    "static/img/Dove.png"
]

# List of Prices
prices = [200, 300, 100, 70, 90, 450, 800, 500]
# Functions

brand_image_map = {
    "Dove": "static/img/Dove.png",
    "COVERGIRL": "static/img/covergirl.png",
    "Suave": "static/img/Suave.png",
    "Love Beauty and Planet": "static/img/love beauty and planet.png",
    "Vaseline": "static/img/Vaseline.png",
    "Rusk": "static/img/Rusk.png",
    "Axe": "static/img/Axe.png",
    "BareMinerals": "static/img/BareMinerals.png"
    # Add more brands and their corresponding image filenames here
}


def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


# Function to search products based on a query
def search_products(product_data, query, top_n=10):
    # Transform the query using the same TF-IDF vectorizer
    query_tfidf = tfidf_vectorizer.transform([query])

    # Compute cosine similarity between the query and product descriptions
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Get indices of the most similar products
    similar_product_indices = cosine_similarities.argsort()[:-top_n - 1:-1]

    # Get the top recommended products
    recommended_products = product_data.iloc[similar_product_indices]

    return recommended_products[['ProdID', 'Name', 'Brand', 'Category', 'ImageURL']].drop_duplicates(subset='ProdID')


def recommend_content_based(product_id, product_data, top_n=15):
    # Get the index of the product
    idx = product_data[product_data['ProdID'] == product_id].index[0]

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the similarity scores for the product with all other products
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N most similar products (excluding the product itself)
    sim_scores = sim_scores[1:top_n + 1]

    # Get the product indices and corresponding similarity scores
    product_indices = [i[0] for i in sim_scores]

    # Fetch the product details for the recommended products
    recommended_products = product_data.iloc[product_indices][
        ['ProdID', 'Name', 'Brand', 'Category', 'ImageURL']].drop_duplicates(subset='ProdID')

    return recommended_products


def user_based_recommendation(user_id, top_n=12):

    product_data['user_id'] = product_data['user_id'].astype(str)
    user_item_matrix = product_data.pivot_table(index='user_id', columns='ProdID', values='Rating', fill_value=0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    user_item_matrix = product_data.pivot_table(index='user_id', columns='ProdID', values='Rating', fill_value=0)



    # if user_id not in user_item_matrix.index:
    #     print(f"user_id {user_id} not found in user_item_matrix")
    # else:
    #     print(f"user_id {user_id} found in user_item_matrix")
    #
    # print("Type of user_id in user_item_matrix:", type(user_item_matrix.index[0]))
    # print("Type of user_id from session:", type(user_id))
    # print("Ratings for user:", user_item_matrix.loc[user_id])
    # Get the list of users that are most similar to the target user
    similar_users = user_similarity_df.loc[str(user_id)].sort_values(ascending=False)[1:].head(top_n)

    # Get ratings of similar users
    similar_users_ratings = user_item_matrix.loc[similar_users.index]

    # Predict ratings for the target user by averaging the ratings from similar users
    predicted_ratings = similar_users_ratings.T.dot(similar_users.values) / similar_users.values.sum()

    # Get the items the user has not rated yet
    user_rated_items = user_item_matrix.loc[str(user_id)] > 0
    unrated_items = predicted_ratings[user_rated_items == False]

    # Sort unrated items by predicted rating
    recommended_items = unrated_items.sort_values(ascending=False).head(top_n)

    # Fetch product details for the recommended items
    recommended_product_details = product_data[product_data['ProdID'].isin(recommended_items.index)][
        ['ProdID', 'Name', 'Brand', 'Category', 'ImageURL']].drop_duplicates(subset='ProdID')

    return recommended_product_details


# Flask route for the index_page (First Page)
@app.route("/")
@app.route('/index')
def indexredirect():

    user_id = session.get('user_id')
    # print(product_data['ProdID'].nunique())
    # print(product_data['user_id'].nunique())


    if user_id:
        # Generate recommendations for the logged-in user
        recommended_products = user_based_recommendation(user_id)
    else:
        # Default case if no user is logged in
        recommended_products = product_data.head(12)  # Fallback: Random products

    random_product_image_urls = [
        brand_image_map.get(product['product_brand'], "static/img/img_1.png")
        # Use a default image if brand is not in the map
        for _, product in trending_products.iterrows()
    ]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', user_id=user_id, trending_products=trending_products.head(9),
                           recommended_products=recommended_products.head(12), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))


# Flask route for the login_page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id")  # Get user ID from the form
        if user_id in user_item_matrix.index:
            session["user_id"] = user_id
            # Redirect to index with user ID as a query parameter or session
            return redirect(url_for("indexredirect", user_id=user_id))
        else:
            error_message = "Please enter a valid user ID."
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
    brand_image_map = {

        "Dove": "static/img/Dove.png",
        "COVERGIRL": "static/img/covergirl.png",
        "Suave": "static/img/Suave.png",
        "Love Beauty and Planet": "static/img/love beauty and planet.png",
        "Vaseline": "static/img/Vaseline.png",
        "Rusk": "static/img/Rusk.png",
        "Axe": "static/img/Axe.png",
        "BareMinerals": "static/img/BareMinerals.png"
        # Add more brands and their corresponding image filenames here
    }

    user_id = session.get("user_id")

    if request.method == 'POST':
        prod = request.form.get('prod')
        content_based_rec = search_products(product_data, prod, top_n=10)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls_1 = [
                brand_image_map.get(product['Brand'], "static/img/img_1.png")
                # Use a default image if brand is not in the map
                for _, product in content_based_rec.iterrows()
            ]
            print(content_based_rec)
            print(random_product_image_urls_1)

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', user_id=user_id, content_based_rec=content_based_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls_1,
                                   random_price=random.choice(price))


@app.route("/View_Similar_Product")
def View_Similar_Product():
    brand_image_map = {

        "Dove": "static/img/Dove.png",
        "COVERGIRL": "static/img/covergirl.png",
        "Suave": "static/img/Suave.png",
        "Love Beauty and Planet": "static/img/love beauty and planet.png",
        "Vaseline": "static/img/Vaseline.png",
        "Rusk": "static/img/Rusk.png",
        "Axe": "static/img/Axe.png",
        "BareMinerals": "static/img/BareMinerals.png"
        # Add more brands and their corresponding image filenames here
    }

    user_id = session.get("user_id")

    product_id = request.args.get('product_id')
    if not product_id:
        # If product_id is missing, redirect or show an error message
        message = "Product ID is missing. Please try again."
        return render_template('main.html', message=message)

        # Call the recommendation function (ensure it handles invalid IDs)
    Recommendations = recommend_content_based(product_id, product_data)

    if Recommendations.empty:
        # Handle the case where there are no recommendations
        message = "No recommendations available for this product."
        return render_template('main.html', message=message)

        # Process the recommendations
    random_product_image_urls_1 = [
        brand_image_map.get(product['Brand'], "static/img/img_1.png")
        for _, product in Recommendations.iterrows()
    ]
    # print(Recommendations)
    # print(random_product_image_urls_1)

    # Random prices (or replace with dynamic data)
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    # Return the rendered template with data
    return render_template(
        'similar_product.html', user_id=user_id,
        recommended_products=Recommendations,
        truncate=truncate,
        random_product_image_urls=random_product_image_urls_1,
        random_price=random.choice(price)
    )


# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
    app.run(debug=True)
