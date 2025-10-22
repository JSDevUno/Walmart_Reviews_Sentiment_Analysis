# Walmart Sentiment Analyzer

A modern web application that analyzes customer reviews from Walmart products using advanced AI and machine learning models.

## 🚀 Features

- **Real-time Sentiment Analysis**: Analyze product reviews using RoBERTa and TF-IDF models
- **Modern React Frontend**: Built with Vite + React for optimal performance
- **Flask API Backend**: RESTful API with CORS support
- **Interactive Dashboard**: Beautiful charts and statistics
- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Theme**: Toggle between themes
- **Real-time Progress**: Live updates during analysis

## 🏗️ Project Structure

```
WALMART_SENTIMENT_ANALYSIS/
├── backend/                    # Flask API Server
│   ├── app.py                 # Main Flask application
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── layout/        # Layout components
│   │   │   ├── analysis/      # Analysis components
│   │   │   └── ui/           # UI components
│   │   ├── pages/            # Page components
│   │   ├── styles/           # CSS styles
│   │   ├── App.jsx           # Main app component
│   │   └── main.jsx          # React entry point
│   ├── index.html            # HTML entry point
│   ├── package.json          # Node.js dependencies
│   └── vite.config.js        # Vite configuration
├── NLP/                      # Machine Learning Models
│   ├── MODEL.py              # Model training
│   ├── inference.py          # Sentiment inference
│   └── train_balance.py      # Balanced training
└── dataset/                  # Training data
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Chrome/Chromium** browser
- **Trained ML model** (see NLP section)

### Backend Setup (Flask API)

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have a trained model:**
   - Place your trained model file (`.pkl`) in the backend directory
   - Or create a `latest_model.txt` file pointing to your model

4. **Start the Flask server:**
   ```bash
   python app.py
   ```
   - Server runs on `http://localhost:5000`

### Frontend Setup (React)

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   - Frontend runs on `http://localhost:3000`
   - Automatically proxies API calls to Flask backend

## 🎯 Usage

1. **Open the application** in your browser at `http://localhost:3000`

2. **Enter a Walmart product URL** in the input field

3. **Click "Analyze Reviews"** to start the analysis

4. **View results** in the modal that appears with:
   - Sentiment distribution charts
   - Sample reviews for each sentiment
   - Confidence scores
   - Average ratings

## 🔧 API Endpoints

### `POST /api/analyze`
Analyze reviews for a Walmart product.

**Request:**
```json
{
  "url": "https://www.walmart.com/ip/product-url",
  "max_reviews": 50
}
```

**Response:**
```json
{
  "session_id": "1234567890.123"
}
```

### `GET /api/status/<session_id>`
Get analysis status and results.

**Response:**
```json
{
  "status": "complete",
  "data": {
    "metadata": {
      "total_reviews": 50,
      "positive_count": 25,
      "negative_count": 15,
      "neutral_count": 10,
      "average_confidence": 0.85
    },
    "samples": {
      "positive": [...],
      "negative": [...],
      "neutral": [...]
    }
  }
}
```

### `GET /api/health`
Health check endpoint.

## 🤖 Machine Learning Models

### Training a Model

1. **Navigate to NLP directory:**
   ```bash
   cd NLP
   ```

2. **Train a basic model:**
   ```bash
   python MODEL.py
   ```

3. **Train with class balancing:**
   ```bash
   python train_balance.py
   ```

4. **Copy the trained model to backend:**
   ```bash
   cp *.pkl ../backend/
   ```

### Model Architecture

- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification**: LinearSVC (Support Vector Machine)
- **Sentiment Labels**: Positive, Negative, Neutral
- **Features**: Review text + title combination

## 🎨 Frontend Components

### Layout Components
- **`Navbar`**: Top navigation with theme toggle
- **`Sidebar`**: Mobile navigation menu
- **`AboutModal`**: Information about the application

### Analysis Components
- **`AnalysisForm`**: URL input and analysis trigger
- **`SentimentChart`**: Visual sentiment distribution
- **`ReviewCard`**: Individual review display
- **`ResultsModal`**: Complete analysis results

### UI Components
- **`Button`**: Reusable button component
- **`Modal`**: Modal dialog wrapper
- **`Spinner`**: Loading indicator

## 🎨 Styling

- **CSS Variables**: Theme management with light/dark mode
- **Responsive Design**: Mobile-first approach
- **Modern UI**: Clean, professional interface
- **Smooth Animations**: Enhanced user experience

## 🚀 Deployment

### Backend Deployment
1. **Install dependencies** on your server
2. **Set up a WSGI server** (e.g., Gunicorn)
3. **Configure reverse proxy** (e.g., Nginx)
4. **Set environment variables** for production

### Frontend Deployment
1. **Build the production bundle:**
   ```bash
   npm run build
   ```
2. **Deploy the `dist/` folder** to your web server
3. **Configure API endpoints** for production backend

## 🔍 Development

### Adding New Features
1. **Frontend**: Add components in `src/components/`
2. **Backend**: Add routes in `app.py`
3. **Styling**: Update `src/styles/globals.css`

### Testing
- **Frontend**: `npm run lint` for code quality
- **Backend**: Add unit tests for API endpoints
- **Integration**: Test full workflow with real URLs

## 📊 Data Flow

1. **User enters Walmart URL**
2. **Frontend sends POST to `/api/analyze`**
3. **Backend starts browser automation**
4. **Selenium scrapes reviews from multiple pages**
5. **ML model analyzes each review sentiment**
6. **Results aggregated and returned**
7. **Frontend displays interactive dashboard**

## 🛡️ Security Considerations

- **CORS enabled** for cross-origin requests
- **Input validation** for URLs and parameters
- **Error handling** for browser automation failures
- **Rate limiting** (consider adding for production)

## 📈 Performance

- **React Virtual DOM** for efficient rendering
- **Component-based architecture** for reusability
- **Lazy loading** for large datasets
- **Optimized bundle** with Vite

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **"Model not loaded" error:**
   - Ensure you have a trained `.pkl` model file
   - Check that the model file is in the backend directory

2. **Browser automation fails:**
   - Update Chrome/Chromium browser
   - Check that ChromeDriver is compatible
   - Ensure no other Chrome instances are running

3. **CORS errors:**
   - Verify Flask-CORS is installed
   - Check that backend is running on port 5000

4. **Frontend build errors:**
   - Clear node_modules and reinstall
   - Check Node.js version compatibility

### Getting Help

- Check the console for error messages
- Verify all dependencies are installed
- Ensure both frontend and backend are running
- Check network connectivity between services
