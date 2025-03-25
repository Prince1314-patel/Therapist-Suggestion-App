# Therapist Suggestion App

## Description
The Therapist Suggestion App is a web-based application built with Streamlit that provides therapeutic suggestions based on user input. Using natural language processing (NLP) and machine learning techniques, the app analyzes the user's input and suggests a relevant therapeutic response from a pre-trained dataset.

## Installation
To run this app locally, follow these steps:

### Prerequisites
Ensure you have Python 3.7 or higher installed on your system.

### Clone the Repository
If the project is hosted on GitHub, clone it using:
```bash
git clone https://github.com/yourusername/therapist-suggestion-app.git
```
(Replace `yourusername` with your actual GitHub username.)

### Install Dependencies
Navigate to the project directory and install the required libraries:
```bash
pip install -r requirements.txt
```

### Download NLTK Data
The app requires specific NLTK resources for text processing. Run the following commands in a Python shell:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Running the App
Once the installation is complete, launch the app using Streamlit:
```bash
streamlit run app.py
```
This will open the app in your default web browser.

## Usage
1. **Input Your Sentence**: Enter a sentence or phrase in the text input box (e.g., "I feel anxious about my job.").
2. **Get a Suggestion**: Click the "Get Suggestion" button to receive a therapeutic response based on your input.
3. **View the Response**: The app will display a suggested response along with a similarity score indicating how closely it matches your input.
   - If no matching response is found, the app will prompt you to try rephrasing your input.

## Project Structure
- `app.py`: The main Streamlit application file that handles user input, processes it, and displays suggestions.
- `train.csv`: The dataset containing pre-processed therapeutic contexts and responses used by the model.
- `requirements.txt`: A list of Python dependencies required to run the app.

## Technologies Used
- **Streamlit**: For building the interactive web application.
- **scikit-learn**: For text vectorization (TF-IDF), dimensionality reduction (LSA), and similarity calculations.
- **NLTK**: For text preprocessing, including tokenization and stopword removal.
- **pandas**: For data manipulation and handling the dataset.

## Contributing
Contributions are welcome! If you'd like to improve the app or report issues, please feel free to submit a pull request or open an issue on the project's GitHub page.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Notes
- **GitHub Link**: Replace `yourusername` in the clone command with your actual GitHub username if the project is hosted on GitHub.
- **License**: The MIT License is used here as a default. If you prefer a different license, update the `LICENSE` file accordingly.
- **NLTK Data**: Make sure to download the necessary NLTK data during setup, as it’s critical for the app’s text processing functionality.

This README file provides all the essential information needed to get started with the Therapist Suggestion App! Feel free to modify or add details as required.

