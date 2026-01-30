"""
Main entry point for the Income Prediction application.

It initializes and launches the GUI for processing socio-economic data,
training the Random Forest model, generating visualizations, and
reviewing model performance.
"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from gui import launch_gui
    launch_gui()