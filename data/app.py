from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras import metrics
import mysql.connector

# Aktifkan eager execution jika belum aktif
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Load model dan preprocessors
model = tf.keras.models.load_model('budget_suggestion_model.h5', custom_objects={'mse': metrics.MeanSquaredError()})
with open('label_encoder.pkl', 'rb') as f:
    le_category = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Koneksi ke database MySQL
def get_db_connection():
    connection = mysql.connector.connect(
        host="34.128.106.108",
        user="root", 
        password="123454321", 
        database="auth_api",
    )
    return connection

# Fungsi untuk preprocessing data
def preprocess_data(data):
    data['category_encoded'] = le_category.transform(data['category'])
    scaled_features = scaler.transform(data[['amount']])
    data['amount_scaled'] = scaled_features[:, 0]
    return data

# Initialize Flask app
app = Flask(__name__)

@app.route('/suggest_budget', methods=["POST"])
def suggest_budget():
    data = request.get_json()
    userId = data.get('userId')
    
    if not userId:
        return jsonify({"error": "User ID is required"}), 400

    # Koneksi ke database dan ambil data transaksi untuk userId tertentu
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT userId, date, amount, category, type
        FROM Transactions
        WHERE userId = %s
    """, (userId,))
    transactions = cursor.fetchall()
    connection.close()

    if not transactions:
        return jsonify({"error": "User not found or no transactions available"}), 404

    # Convert data transaksi menjadi DataFrame
    dataset = pd.DataFrame(transactions)
    dataset['date'] = pd.to_datetime(dataset['date'])

    # Filter hanya data Expense
    user_data = dataset[dataset['type'] == 'Expense']

    # Preprocessing data
    user_data['month'] = user_data['date'].dt.month
    user_data['day'] = user_data['date'].dt.day
    user_data = preprocess_data(user_data)

    if user_data.empty:
        return jsonify({"error": "No data available for the specified user"}), 404

    # Menghitung total pengeluaran pengguna
    total_expense = user_data['amount'].sum()
    num_months = user_data['month'].nunique()

    # Hitung rekomendasi per kategori
    recommendations = []
    if total_expense > 0:
        category_distribution = (
            user_data.groupby('category_encoded')['amount'].sum() / total_expense
        )
        monthly_average = total_expense / num_months
        recommendations = [
            {
                "category": le_category.inverse_transform([category])[0],
                "budget_recommendation": round(percent * monthly_average, 2),
            }
            for category, percent in category_distribution.items()
        ]

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)