from setuptools import setup, find_packages

__version__ = "0.0.1"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="imdb_movie_ratings",
    version=__version__,
    description="Predicting a movie's average rating on IMDb.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LeoHunterSlays/imdb-movie-ratings.git",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    author="Leo Hunter",
    python_requires=">=3.8",
    setup_requires=[
        "numpy",
        "scipy",
        "pandas",
        "xgboost",
        "shap",
        "matplotlib",
        "seaborn",
        "sklearn",
    ],
    author_email="sf.bj.la.ny@gmail.com",
)
