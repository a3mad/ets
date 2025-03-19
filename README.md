
# Customer Segmentation Project

## Installation and Setup Instructions

### Prerequisites
Before running the project, ensure you have the following installed on your system:
- **Python 3.8+**
- **pip** (Python package manager)
- A virtual environment tool such as `venv` or `virtualenv` (recommended)
- **Git** (to clone the repository)

### Steps to Install and Run the Project

1. **Clone the Repository**  
   Clone the project repository to your local machine:

   ```bash
   git clone https://github.com/a3mad/segmentation-ml.git
   ```
   
2. **Create and Activate a Virtual Environment**  
   Create and activate a virtual environment to isolate project dependencies.

   **On Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   **On macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**  
   Install the required Python libraries from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Set Flask Environment Variables**  
   Set the necessary environment variables to run the Flask app.

   **On Windows:**
   ```bash
   set FLASK_APP=run.py
   set FLASK_ENV=development
   ```
   **On macOS/Linux:**
   ```bash
   export FLASK_APP=run.py
   export FLASK_ENV=development
   ```

5. **Run the Application**  
   Start the Flask development server:
   ```bash
   flask run
   ```


6. **Interact with the Application**  
   - Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).  
   - Follow these steps in the application:
     - Select a segmentation type.
     - Upload your dataset.
     - Match required and additional columns.
     - Review the selected columns.
     - Run the segmentation process and view results.

7. **Optional: Debug Mode**  
   To limit dataset rows during debugging:
   - Open `config.py`.
   - Set `DEBUG_MODE = True`.

8. **Deactivate the Virtual Environment**  
   When you're done, deactivate the virtual environment: 
   ```bash
   deactivate
   ```

### Troubleshooting
- **Dependencies Not Installing:** Ensure you are using the correct version of Python and pip.
- **Server Not Starting:** Verify that `FLASK_APP` is set to `run.py`.
- **File Upload Issues:** Ensure your dataset file is in CSV format.