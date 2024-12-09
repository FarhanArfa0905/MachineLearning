from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
# from tensorflow.keras.models import load_model
import pickle

# Load model dan preprocessors
model = tf.keras.models.load_model('budget_suggestion_model.h5', compile=False)
with open('label_encoder.pkl', 'rb') as file:
    le_category = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/suggest_budget', methods=['POST'])
def suggest_budget():
    # Ambil data dari request
    data = request.get_json()
    user_id = data.get('user_id')
    
    # Load dataset
    dataset = pd.read_csv('Transaction_Data.csv')
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Filter data untuk user tertentu
    user_data = dataset[dataset['user_id'] == user_id]
    if user_data.empty:
        return jsonify({"error": "User not found"}), 404

    # Filter hanya data Expense
    user_data = user_data[user_data['type'] == 'Expense']

    # Preprocessing data
    user_data['month'] = user_data['date'].dt.month
    user_data['day'] = user_data['date'].dt.day
    user_data['category_encoded'] = le_category.transform(user_data['category'])

    #Validasi ada user
    if user_data.empty:
        return jsonify({"error": "No data available for the specified user"}), 404

    # Menghitung total pengeluaran pengguna
    total_expense = user_data['amount'].sum()
    num_months = user_data['month'].nunique()  # Hitung jumlah bulan yang unik dalam data

    # Hitung rekomendasi per kategori
    recommendations = []
    if total_expense > 0:
        category_distribution = (
            user_data.groupby('category_encoded')['amount'].sum() / total_expense
        )
        monthly_average = total_expense / num_months  # Rata-rata pengeluaran per bulan
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
