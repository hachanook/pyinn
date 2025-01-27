from setuptools import setup, find_packages

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name='pyinn',
    version='0.1.3',
    author='Chanwook Park',
    author_email='chanwookpark2024@u.northwestern.edu',
    description='Interpolating Neural Networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)