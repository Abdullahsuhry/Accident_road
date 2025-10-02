# Road Accident Severity Prediction App

A comprehensive web application that predicts road accident severity using machine learning and provides user authentication.

## Features

- **User Authentication**: Secure registration and login system
- **Accident Severity Prediction**: AI-powered prediction using machine learning
- **Modern UI**: Responsive design with smooth animations
- **Real-time Notifications**: User feedback system
- **Data Analytics**: Comprehensive road safety analysis

## Technology Stack

### Frontend
- HTML5
- CSS3 (with modern features like backdrop-filter, grid, flexbox)
- JavaScript (ES6+)
- Font Awesome icons

### Backend
- Python Flask
- SQLite database
- JWT authentication
- Scikit-learn for machine learning
- Pandas for data processing

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask application:**
   ```bash
   python app.py
   ```

   The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Open the HTML file:**
   - Simply open `index.html` in your web browser
   - Or serve it using a local web server:
     ```bash
     # Using Python's built-in server
     python -m http.server 8000
     ```
   - Then visit `http://localhost:8000`

## API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login

### Prediction
- `POST /api/predict` - Predict accident severity (requires authentication)

### Profile
- `GET /api/profile` - Get user profile (requires authentication)

### Health Check
- `GET /api/health` - API health status

## Usage

### 1. User Registration/Login
- Click "Register" to create a new account
- Click "Login" to sign in with existing credentials
- Authentication is required for prediction features

### 2. Accident Severity Prediction
- Navigate to the prediction section
- Fill in the required information:
  - Age
  - Gender
  - Vehicle Type
  - Weather Condition
  - Road Type
  - Speed
- Click "Predict Severity" to get the prediction

### 3. Prediction Results
The system will display:
- Severity level (Low/Medium/High)
- Confidence percentage
- Detailed analysis

## Machine Learning Model

The application uses a Random Forest Classifier trained on synthetic accident data. The model considers:

- **Demographic factors**: Age, gender
- **Vehicle factors**: Type of vehicle
- **Environmental factors**: Weather conditions, road type
- **Behavioral factors**: Speed

## File Structure

```
log/
├── index.html          # Main HTML file
├── styles.css          # CSS styling
├── script.js           # JavaScript functionality
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── users.db            # SQLite database (created automatically)
```

## Security Features

- Password hashing using Werkzeug
- JWT token-based authentication
- CORS protection
- Input validation and sanitization

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For support or questions, please open an issue in the repository.

