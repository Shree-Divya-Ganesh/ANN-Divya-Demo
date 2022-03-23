from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='src',
      version='0.0.1',
      description='ANN-Implementation-Divya-Oct',
      author='DivyaGanesh',
      author_email='shreedivyaganesh@gmail.com',
      url='https://github.com/Shree-Divya-Ganesh/ANN-Divya-Demo',
      packages=['src'],
      python_requires = ">= 3.7",
      install_requires=[
          "tensorflow",
          "matplotlib",
          "seaborn",
          "pandas",
          "numpy"
      ]
     )