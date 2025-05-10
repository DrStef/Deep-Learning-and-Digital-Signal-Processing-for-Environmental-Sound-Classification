from setuptools import setup, find_packages

setup(
    name="esc10_sound_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'librosa',
        'pywt',
        'tensorflow',
        'scikit-learn',
        'seaborn',
        'scikit-image',
        'pytest',
        'flake8'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="ESC-10 sound classification using CWT, Mel-spectrograms, and CNN",
    license="CC BY",
    keywords="signal processing, audio classification, deep learning",
    url="https://github.com/yourusername/esc10_sound_classification"
)