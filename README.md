# 💰 Income Predictor

This project is a desktop application that predicts whether an individual's income is greater than $50K based on socio-economic data. It provides a user-friendly interface to load data, train a machine learning model, and visualize the data.

## ✨ Features

*   **Load and Clean Data**: Load the "adult" dataset and preprocess it for analysis.
*   **Train Machine Learning Model**: Train a Random Forest Classifier to predict income level.
*   **Data Visualization**: Generate various plots to explore the dataset, including:
    *   Age and hour-per-week distributions
    *   Education level and race pie charts
    *   Workclass, income, marital status, and gender distributions
    *   Age vs. income and education vs. income relationships
    *   Occupation treemap and heatmap
    *   Correlation heatmap
*   **Configuration**: Customize the application's appearance through a simple configuration file.

## 🚀 How to Use

1.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment**:
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Application**:
    ```bash
    python income_predictor.py
    ```
5.  **Use the Application**:
    *   **Data Page**:
        1.  Click "Load & Clean Data" to load the dataset.
        2.  Click "Inspect Data" to see a summary of the data.
        3.  Click "Split X & y" to separate features and the target variable.
        4.  Click "Train Model" to train the Random Forest model.
        5.  Click "Export Report" to save the training report.
    *   **Visualization Page**:
        1.  Click "Load & Clean Data".
        2.  Select a plot from the dropdown menu.
        3.  Click "Generate Plot" to display the visualization.
    *   **Configuration Page**:
        1.  Modify the settings as desired.
        2.  Click "Save Configuration" and the application will restart with the new settings.

## 📁 Project Structure

```
Work/
├── Data/
│   └── adult.csv
├── Graphics/
├── Library/
│   ├── data_utils.py
│   └── plot_utils.py
├── Notes/
│   ├── DEVELOPER's Guide.pdf
│   └── USER MANUAL.pdf
├── Output/
│   └── Training_report.txt
└── Scripts/
    ├── config.ini
    ├── gui.py
    ├── income_predictor.py
    ├── model_utils.py
    └── requirements.txt
```

## 🛠️ Dependencies

The project's dependencies are listed in the `requirements.txt` file. The main dependencies are:

*   `pandas`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `squarify`

## ⚙️ Configuration

The application can be configured by editing the `config.ini` file in the `Scripts` directory. The following settings can be changed:

*   `window_width`, `window_height`
*   `bg_color`
*   `font_family`, `font_size`, `sidebar_font_size`
*   `report_button_color`, `button_color`, `export_button_color`
