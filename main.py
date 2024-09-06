import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import linear_kernel
from tensorflow.keras.utils import img_to_array
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import pyodbc

# Load data from the database
app = Flask(__name__)
app.config['SQL_SERVER'] = 'DESKTOP-9LCUUH1'
app.config['DATABASE'] = 'FashionRecommendation'
app.config['USERNAME'] = 'sa'
app.config['PASSWORD'] = '123qwe'
app.config['DRIVER'] = '{ODBC Driver 17 for SQL Server}'

# Connect to the database
def get_db_connection():
    conn_str = f'DRIVER={app.config["DRIVER"]};SERVER={app.config["SQL_SERVER"]};DATABASE={app.config["DATABASE"]};UID={app.config["USERNAME"]};PWD={app.config["PASSWORD"]}'
    return pyodbc.connect(conn_str)

# Function to fetch and process images from URLs
def fetch_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((100, 100))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Fashion recommendation class
class FashionRecommendations:
    def __init__(self):
        self.vgg16_model = self.load_model()
        self.df_embeddings = self.load_embeddings()

    def load_model(self):
        model_save_path = 'vgg16_model.h5'
        if os.path.exists(model_save_path):
            print("Loading pre-trained model from", model_save_path)
            vgg16_model = keras.models.load_model(model_save_path, compile=False)
            print("Model loaded successfully.")
        else:
            print("Training a new VGG16 model")
            vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(100, 100, 3))
            vgg16.trainable = False
            vgg16_model = keras.Sequential([vgg16, GlobalMaxPooling2D()])
            vgg16_model.save(model_save_path)
            print("Model trained and saved successfully.")
        return vgg16_model

    def load_embeddings(self):
        embeddings_path = 'embeddings.csv'
        if os.path.exists(embeddings_path):
            df = pd.read_csv(embeddings_path)
            embedding_columns = [col for col in df.columns if col.startswith('embedding_')]
            df[embedding_columns] = df[embedding_columns].astype(float)
            return df
        else:
            db_conn = get_db_connection()
            df = pd.read_sql("SELECT ProductID, ImageURL FROM products", db_conn)
            df['embeddings'] = df['ImageURL'].apply(lambda url: self.get_image_embedding(url))

            embedding_df = pd.DataFrame(df['embeddings'].to_list(),
                                        columns=[f'embedding_{i}' for i in range(len(df['embeddings'][0]))])
            df = pd.concat([df, embedding_df], axis=1)
            df.drop(columns=['embeddings'], inplace=True)

            df.to_csv(embeddings_path, index=False)
            return df

    def get_image_embedding(self, img_url):
        img = fetch_image_from_url(img_url)
        embedding = self.vgg16_model.predict(img)
        return embedding.flatten().tolist()

    def update_embeddings(self, product_id, img_url):
        new_embedding = self.get_image_embedding(img_url)
        new_entry = pd.DataFrame([[product_id, img_url] + new_embedding],
                                 columns=['ProductID', 'ImageURL'] + [f'embedding_{i}' for i in
                                                                      range(len(new_embedding))])
        self.df_embeddings = pd.concat([self.df_embeddings, new_entry], ignore_index=True)
        self.df_embeddings.to_csv('embeddings.csv', index=False)
        # Reload embeddings to ensure the new product is included
        self.df_embeddings = self.load_embeddings()

    def get_similarity(self, img_url):
        sample_embedding = self.get_image_embedding(img_url)
        similarity_scores = linear_kernel([sample_embedding], self.df_embeddings.iloc[:, 2:].values)
        return similarity_scores.flatten()

    def get_recommendations(self, img_urls):
        recommendations = []
        for img_url in img_urls:
            similarity_scores = self.get_similarity(img_url)
            top_indices = similarity_scores.argsort()[::-1][:20]  # Get top 20 similar items
            top_products = self.df_embeddings.iloc[top_indices]

            # Introduce randomness by shuffling
            top_products = top_products.sample(frac=1).reset_index(drop=True)

            # Pick the top 5 after shuffling
            top_products = top_products.head(5)

            db_conn = get_db_connection()
            cursor = db_conn.cursor()

            for _, row in top_products.iterrows():
                try:
                    product_id = row['ProductID']
                    cursor.execute("""
                        SELECT ProductID, ProductDisplayName, ImageURL, Category, Price, Description, Gender, 
                               MasterCategory, SubCategory, ArticleType, BaseColour, Season, Year, usage
                        FROM products WHERE ProductID = ?
                    """, (product_id,))

                    product = cursor.fetchone()
                    if product:
                        recommendations.append({
                            'ProductID': product[0],
                            'ProductDisplayName': product[1],
                            'ImageURL': product[2],
                            'Category': product[3],
                            'Price': float(product[4]) if product[4] is not None else None,
                            'Description': product[5],
                            'Gender': product[6],
                            'MasterCategory': product[7],
                            'SubCategory': product[8],
                            'ArticleType': product[9],
                            'BaseColour': product[10],
                            'Season': product[11],
                            'Year': product[12],
                            'usage': product[13]
                        })
                except Exception as e:
                    print(f"Error processing product ID {row['ProductID']}: {e}")

            cursor.close()
            db_conn.close()

        return recommendations


# Instantiate the fashion recommendation object
fashion_rec = FashionRecommendations()

# API endpoint for getting recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations_api():
    data = request.json
    user_id = data.get('User_Id')
    print(user_id)
    if not user_id:
        return jsonify({'error': 'No user ID provided'}), 400

    db_conn = get_db_connection()
    cursor = db_conn.cursor()

    # Fetch all image URLs for the given UserID
    cursor.execute("""
        SELECT TOP 4 Image_Url 
        FROM actions 
        WHERE User_Id = ? 
        ORDER BY Id DESC 
    """, (user_id,))
    actions = cursor.fetchall()

    if not actions:
        return jsonify({'error': 'No image URLs found for the given UserID'}), 404

    # Extract image URLs from the fetched records
    img_urls = [action[0] for action in actions]

    # Get recommendations for all image URLs
    recommendations = fashion_rec.get_recommendations(img_urls)

    cursor.close()
    db_conn.close()

    return jsonify({'recommendations': recommendations}), 200



# API endpoint for inserting a new product
@app.route('/products', methods=['POST'])
def add_product():
    data = request.json
    ImageURL = data.get('ImageURL')
    Category = data.get('Category')
    Price = data.get('Price')
    Description = data.get('Description')
    Gender = data.get('Gender')
    MasterCategory = data.get('MasterCategory')
    SubCategory = data.get('SubCategory')
    ArticleType = data.get('ArticleType')
    BaseColour = data.get('BaseColour')
    Season = data.get('Season')
    Year = data.get('Year')
    usage = data.get('usage')
    ProductDisplayName = data.get('ProductDisplayName')

    conn = get_db_connection()
    cursor = conn.cursor()

    # Perform the INSERT operation
    cursor.execute(
        """
        INSERT INTO products (ImageURL, Category, Price, Description, Gender, 
                              MasterCategory, SubCategory, ArticleType, BaseColour, Season, Year, usage, ProductDisplayName)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? );
        """,
        (ImageURL, Category, Price, Description, Gender, MasterCategory, SubCategory, ArticleType, BaseColour,
         Season, Year, usage, ProductDisplayName)
    )


    cursor.execute("SELECT SCOPE_IDENTITY()")
    ProductID = cursor.fetchone()[0]

    conn.commit()
    cursor.close()
    conn.close()

    # Update the embeddings with the new product
    fashion_rec.update_embeddings(ProductID, ImageURL)

    return jsonify({'message': 'Product added successfully', 'ProductID': ProductID}), 201


# API endpoint for inserting a new user
@app.route('/users', methods=['POST'])
def add_user():
    data = request.json
    username = data.get('username')
    age = data.get('age')
    gender = data.get('gender')
    style_preference = data.get('style_preference')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO users (Username, Age, Gender, StylePreference)
        VALUES (?, ?, ?, ?)
        """,
        (username, age, gender, style_preference)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'message': 'User added successfully'}), 201

# API endpoint for adding a new action
@app.route('/actions', methods=['POST'])
def add_action():
    data = request.json
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    action_type = data.get('action_type')
    image_url = data.get('image_url')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO actions (User_Id, Product_Id, Action_Type, Image_Url)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, product_id, action_type, image_url)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'message': 'Action added successfully'}), 201


# API endpoint for adding a product to wishlist
@app.route('/wishlist', methods=['POST'])
def add_to_wishlist():
    data = request.json
    user_id = data.get('user_id')
    product_id = data.get('product_id')

    if not user_id or not product_id:
        return jsonify({'error': 'User ID and Product ID are required'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO wishlist (User_Id, Product_Id)
            VALUES (?, ?)
            """,
            (user_id, product_id)
        )
        conn.commit()
    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({'error': 'Failed to add to wishlist'}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({'message': 'Product added to wishlist successfully'}), 201

if __name__ == '__main__':
    app.run(debug=True)
