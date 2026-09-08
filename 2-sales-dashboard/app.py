from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app)

# Generate sample data
def generate_sample_data():
    data = []
    start_date = datetime.now() - timedelta(days=90)
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    
    for i in range(500):
        data.append({
            'date': start_date + timedelta(days=random.randint(0, 90)),
            'region': random.choice(regions),
            'product': random.choice(products),
            'revenue': random.randint(100, 5000),
            'quantity': random.randint(1, 50),
            'customer_id': f'CUST{random.randint(1000, 9999)}'
        })
    
    return pd.DataFrame(data)

df = generate_sample_data()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    total_revenue = df['revenue'].sum()
    total_sales = len(df)
    avg_order_value = df['revenue'].mean()
    unique_customers = df['customer_id'].nunique()
    
    return jsonify({
        'total_revenue': round(total_revenue, 2),
        'total_sales': total_sales,
        'avg_order_value': round(avg_order_value, 2),
        'unique_customers': unique_customers,
        'conversion_rate': round((unique_customers / total_sales) * 100, 2)
    })

@app.route('/api/sales-by-date')
def sales_by_date():
    daily_sales = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
    daily_sales.columns = ['date', 'revenue']
    
    return jsonify({
        'labels': daily_sales['date'].astype(str).tolist(),
        'data': daily_sales['revenue'].tolist()
    })

@app.route('/api/sales-by-region')
def sales_by_region():
    region_sales = df.groupby('region')['revenue'].sum().reset_index()
    
    return jsonify({
        'labels': region_sales['region'].tolist(),
        'data': region_sales['revenue'].tolist()
    })

@app.route('/api/top-products')
def top_products():
    top_prod = df.groupby('product')['revenue'].sum().nlargest(5).reset_index()
    
    return jsonify({
        'labels': top_prod['product'].tolist(),
        'data': top_prod['revenue'].tolist()
    })

@app.route('/api/export')
def export_csv():
    csv_data = df.to_csv(index=False)
    return csv_data, 200, {'Content-Disposition': 'attachment; filename=sales_data.csv'}

@app.route('/api/sales')
def get_sales():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    start = (page - 1) * per_page
    end = start + per_page
    
    sales_data = df[start:end].to_dict('records')
    for record in sales_data:
        record['date'] = str(record['date'])
    
    return jsonify({
        'data': sales_data,
        'total': len(df),
        'page': page,
        'per_page': per_page
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)