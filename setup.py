# TODO make this a real package.

# from setuptools import setup
import subprocess
import sys

# with open("README.md", "r", encoding="utf-8") as readme_file:
#     readme = readme_file.read()

# requirements = [
#     "transformers",
#     "datasets",
#     "pandas",
#     "pyyaml",
#     "jinja2",
#     "torch",
#     "black",
#     "jupyter",
#     "tokreate",
#     "evaluate",
#     "nltk"
# ]

# setup(
#     name="science_adapt",
#     version="0.1",
#     install_requires=requirements
# )


def install_packages():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )


def download_nltk_data():
    import nltk

    nltk.download("punkt")
    nltk.download("stopwords")


if __name__ == "__main__":
    install_packages()
    download_nltk_data()
