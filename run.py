# run.py
import subprocess
import webbrowser
import time
import os

def main():
    # Get the path to the app.py file
    # This ensures it works correctly when packaged by PyInstaller
    app_file = os.path.join(os.path.dirname(__file__), "complete_code.py")

    # Command to run the Streamlit server
    command = ["streamlit", "run", app_file, "--server.headless", "true", "--server.port", "8501"]

    # Start the Streamlit server as a background process
    server_process = subprocess.Popen(command)

    # Give the server a few seconds to start up
    time.sleep(5)

    # Open the web browser to the Streamlit app's URL
    webbrowser.open("http://localhost:8501")

    # Keep the launcher running until the server process is terminated
    try:
        server_process.wait()
    except KeyboardInterrupt:
        server_process.terminate()

if __name__ == "__main__":
    main()