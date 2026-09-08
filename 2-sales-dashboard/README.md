# Sales Analytics Dashboard

A comprehensive sales analytics dashboard with real-time metrics, interactive charts, and data visualization.

## Features
- 📊 Real-time sales metrics
- 📈 Interactive charts (Chart.js)
- 🎯 Revenue & performance tracking
- 🔍 Filter & drill-down capabilities
- 📉 Trend analysis
- 💾 CSV export

## Tech Stack
- Python 3.9+
- Flask
- Pandas
- Chart.js
- Bootstrap 5

## Setup

```bash
cd 2-sales-dashboard
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5001`

## Features

### Dashboard Metrics
- Total Revenue
- Average Order Value
- Sales Count
- Customer Acquisition
- Conversion Rate

### Charts
- Sales over time
- Revenue by region
- Top products
- Customer segments
- Monthly trends

## API Endpoints

```
GET /api/metrics - Get all metrics
GET /api/sales - Get sales data
GET /api/export - Export to CSV
```

## License
MIT