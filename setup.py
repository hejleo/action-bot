from setuptools import setup, find_packages

setup(
    name="incar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'Flask==2.0.1',
        'Werkzeug==2.0.3',
        'flask-cors==3.0.10',
        'urllib3==1.26.6',
        'requests==2.31.0',
        'transformers',
        'sentence-transformers',
        'torch',
        'sentencepiece',
        'numpy==1.24.3',
        'pandas==2.0.3',
        'python-dotenv==1.0.0',
        'tqdm==4.66.1',
        'jsonschema==4.20.0',
        'accelerate==0.27.2'
    ],
) 