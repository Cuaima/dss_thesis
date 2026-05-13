from setuptools import setup, find_packages
setup(
    name="custom_text_anonymizer",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
