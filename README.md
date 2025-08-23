# 🎬 Movie Revenue Predictor

A machine learning application that predicts movie revenue based on various features like budget, popularity, runtime, and more. Built with Streamlit and trained on movie dataset analysis.

## 🚀 Features

- **Revenue Prediction**: Predict movie revenue using trained machine learning models
- **Interactive Web Interface**: User-friendly Streamlit application
- **Data Analysis**: Explore movie dataset statistics and visualizations
- **Model Information**: Detailed information about the trained models and their performance

## 📊 Model Performance

Based on the analysis in `movie_predict.ipynb`:

| Model | Train R² | Test R² |
|-------|----------|---------|
| Random Forest | 0.8575 | 0.8169 |
| Gradient Boosting | 0.8910 | 0.8200 |
| Linear Regression (Polynomial) | 0.8158 | 0.8020 |

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone <repository-url>
cd movie-revenue-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Model Files

Ensure you have the following model files in your project directory:
- `ValueClassifier` - Trained Random Forest model
- `encode` - Categorical encoder
- `scaler` - Feature scaler
- `outlier_bounds` - Outlier detection bounds

These files are generated when you run the `movie_predict.ipynb` notebook.

### 4. Prepare Data Files

Ensure you have the following CSV files:
- `movie_details.csv` - Main movie dataset
- `companies.csv` - Company information
- `countries.csv` - Country information
- `genres.csv` - Genre information
- `langs.csv` - Language information

## 🚀 Running the Application

### Start the Streamlit App

```bash
streamlit run deploymen.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📱 Application Features

### 🏠 Home Page
- Input form for movie details
- Real-time revenue prediction
- ROI calculation
- Formatted output (B/M for billions/millions)

### 📊 Data Analysis Page
- Dataset statistics
- Sample data display
- Generated visualizations
- Interactive data exploration

### 🤖 Model Info Page
- Model details and parameters
- Feature information
- Preprocessing steps
- Performance metrics

## 🔧 Model Architecture

### Features Used
1. **Company Name** - Production company
2. **Budget** - Movie budget in dollars
3. **Original Language** - Movie's original language
4. **Popularity** - Popularity score
5. **Runtime** - Movie duration in minutes
6. **Vote Average** - Average user rating (0-10)
7. **Vote Count** - Number of votes
8. **Country** - Production country
9. **Belongs to Collection** - Part of a movie series
10. **Country of Company** - Company's country
11. **Release Year** - Year of release
12. **Genre Count** - Number of genres

### Preprocessing Steps
1. **Categorical Encoding**: TargetEncoder for categorical variables
2. **Feature Scaling**: StandardScaler for numerical features
3. **Outlier Handling**: IQR method for outlier detection and capping
4. **Data Cleaning**: Handling missing values and duplicates

## 📁 Project Structure

```
movie-revenue-predictor/
├── deploymen.py          # Main Streamlit application
├── movie_predict.ipynb   # Jupyter notebook with model training
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── ValueClassifier      # Trained model file
├── encode              # Encoder file
├── scaler              # Scaler file
├── outlier_bounds      # Outlier bounds file
├── *.csv               # Data files
└── *.png               # Generated visualizations
```

## 🔍 Data Sources

The application uses movie data including:
- Movie details (budget, revenue, ratings)
- Company information
- Geographic data
- Genre classifications
- Language information

## 🚀 Deployment Options

### Local Development
```bash
streamlit run deploymen.py
```

### Production Deployment
For production deployment, consider:
- Using Streamlit Cloud
- Docker containerization
- Cloud platforms (AWS, GCP, Azure)

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "deploymen.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure all model files are in the project directory
   - Run the notebook to generate missing files

2. **Data files not found**
   - Check if all CSV files are present
   - Verify file paths and names

3. **Dependencies issues**
   - Update pip: `pip install --upgrade pip`
   - Install requirements: `pip install -r requirements.txt`

### Error Messages

- **"Model not loaded"**: Check model files exist
- **"Data files not found"**: Verify CSV files are present
- **"Encoder error"**: Ensure encoder file matches the model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing web framework
- Scikit-learn for machine learning tools
- Movie dataset contributors
- Open source community

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the notebook for implementation details

---

**Note**: This application requires the trained model files and data files to function properly. Make sure to run the notebook first to generate the necessary model artifacts.


