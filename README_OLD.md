# 🔍 Advanced Crime Analytics Dashboard

A comprehensive data science project analyzing Missing Persons and Juvenile Crimes data across India (2017-2022) using advanced machine learning and deep learning techniques.

## 📊 Project Overview

This project implements a full-stack data science solution featuring:
- **Exploratory Data Analysis (EDA)** with interactive visualizations
- **Regression Models** for predicting missing persons
- **Classification Models** for categorizing crime levels
- **Clustering Analysis** for discovering state-level patterns
- **Deep Learning** using Neural Networks
- **Interactive Dashboard** built with Streamlit

## 🎯 Datasets

1. **Missing Persons Dataset** (2017-2022)
   - 5,319 records across 751 districts
   - 36 Indian states
   - Gender and age-group breakdowns

2. **Juvenile Crimes Dataset** (2017-2022)
   - 5,322 records with 117 crime types
   - District-wise crime statistics
   - Categorized by crime severity

## 🔬 Machine Learning Techniques Implemented

### Regression (Predicting Missing Persons)
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

### Classification (High/Low Crime Districts)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

### Clustering (State-level Patterns)
- K-Means Clustering
- Agglomerative Clustering
- DBSCAN
- PCA for dimensionality reduction

### Deep Learning
- Custom Neural Network Architectures
- Regression Models with Dropout
- Classification Models
- Early Stopping & Learning Rate Scheduling

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone or download the project files**
```bash
# All files should be in the same directory
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Ensure data files are present**
Make sure these CSV files are in the same directory:
- `districtwise-missing-persons-2017-2022.csv`
- `districtwise-ipc-crime-by-juveniles-2017-onwards.csv`

### Running the Application

1. **Start the Streamlit server**
```bash
streamlit run streamlit_app.py
```

2. **Access the dashboard**
The application will automatically open in your browser at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

## 📱 Using the Dashboard

### Navigation
The dashboard is organized into sections accessible from the sidebar:

1. **📊 Overview & EDA**: Dataset information and key statistics
2. **📈 Missing Persons Analysis**: Temporal trends and demographic breakdowns
3. **🔪 Juvenile Crimes Analysis**: Crime patterns and category analysis
4. **🔄 Combined Analysis**: Correlation and comparative insights
5. **🤖 Machine Learning - Regression**: Train and compare regression models
6. **🎯 Machine Learning - Classification**: Train classification models
7. **🌐 Machine Learning - Clustering**: Discover state-level patterns
8. **🧠 Deep Learning**: Neural network implementations
9. **📑 Summary & Insights**: Project overview and key findings

### Features

#### Interactive Visualizations
- **Plotly Charts**: Hover for detailed information
- **Temporal Analysis**: Year-over-year trends
- **Geographic Distributions**: State and district-level comparisons
- **Correlation Analysis**: Relationship between datasets

#### Model Training
- **One-Click Training**: Train multiple models with single button click
- **Performance Metrics**: R², RMSE, MAE for regression; Accuracy for classification
- **Model Comparison**: Side-by-side performance evaluation
- **Best Model Selection**: Automatic identification of top performer

#### Clustering Analysis
- **Adjustable Clusters**: Use slider to change number of clusters
- **Silhouette Scores**: Quality metrics for clustering
- **Pattern Discovery**: Identify state-level crime patterns

#### Deep Learning
- **Customizable Parameters**: Adjust epochs and batch size
- **Training Visualization**: Real-time loss and metric plots
- **Neural Architecture**: Multi-layer networks with dropout

## 📁 Project Structure

```
project/
│
├── streamlit_app.py              # Main Streamlit application
├── data_preprocessing.py         # Data cleaning and feature engineering
├── ml_models.py                  # Machine learning implementations
├── deep_learning.py              # Deep learning models
├── requirements.txt              # Package dependencies
├── README.md                     # This file
│
├── districtwise-missing-persons-2017-2022.csv
└── districtwise-ipc-crime-by-juveniles-2017-onwards.csv
```

## 🔍 Key Insights

### Missing Persons
- Youth (14-30 age group) represents highest proportion of missing cases
- Male missing persons outnumber female cases
- Significant state-wise variation in missing persons rates
- Urban districts show higher absolute numbers

### Juvenile Crimes
- Property crimes most prevalent across all states
- Violent and sexual crimes show concerning trends in specific regions
- Clear clustering of states into distinct crime pattern groups
- Temporal variations suggest seasonal influences

### Combined Analysis
- Moderate positive correlation between missing persons and crime rates
- High-crime districts often coincide with high missing persons rates
- Geographic and socioeconomic factors play significant roles

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow, Keras
- **Visualization**: Plotly, Seaborn, Matplotlib

## 📈 Model Performance

### Regression Models
- **Best Performer**: XGBoost/LightGBM (typically)
- **R² Score**: 0.85-0.95 (varies by dataset)
- **RMSE**: Context-dependent on scale

### Classification Models
- **Best Performer**: Random Forest/XGBoost
- **Accuracy**: 75-90% (depends on threshold)
- **Cross-validation**: Implemented for robustness

### Clustering
- **Optimal Clusters**: 5-7 (based on silhouette score)
- **Pattern Discovery**: Distinct state groupings identified

### Deep Learning
- **Architecture**: 3-layer dense networks with dropout
- **Performance**: Competitive with traditional ML
- **Training Time**: 50-100 epochs typical

## 🎓 Learning Outcomes

This project demonstrates:
1. **End-to-end ML workflow**: From data to deployment
2. **Multiple techniques**: Regression, classification, clustering, deep learning
3. **Feature engineering**: Creating meaningful derived features
4. **Model evaluation**: Comprehensive performance metrics
5. **Visualization**: Interactive, professional dashboards
6. **Production deployment**: Web-based application

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
```bash
# Reinstall packages
pip install --upgrade -r requirements.txt
```

2. **Data File Not Found**
```
Ensure CSV files are in the same directory as streamlit_app.py
```

3. **Port Already in Use**
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

4. **TensorFlow Not Available**
```bash
# Install TensorFlow separately if needed
pip install tensorflow
```

## 📝 Future Enhancements

- [ ] Time series forecasting (ARIMA, Prophet, LSTM)
- [ ] Interactive geographic maps
- [ ] Real-time prediction API
- [ ] Advanced NLP for crime descriptions
- [ ] Explainable AI (SHAP, LIME)
- [ ] Model deployment with Docker
- [ ] Automated retraining pipeline
- [ ] Mobile-responsive design

## 👨‍💻 Development

This project was developed as a comprehensive demonstration of:
- Professional data science workflow
- Modern ML/DL techniques
- Interactive visualization
- Production-ready deployment

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback about this project, please open an issue or contact the developer.

---

**Note**: This is a demonstration project using real Indian crime statistics data. The analysis and models are for educational purposes and should not be used as the sole basis for policy decisions.
