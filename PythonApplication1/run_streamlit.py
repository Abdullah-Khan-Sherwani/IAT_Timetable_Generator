# run_streamlit.py
import os, subprocess

# jump into the folder where your main app lives
os.chdir(os.path.dirname(__file__))

# launches Streamlit exactly as you would in a console
subprocess.run(
    ["python", "-m", "streamlit", "run", "PythonApplication1.py"],
    check=True
)