from setuptools import setup, find_packages

setup(
    name='my_deep_learning',
    version='0.0.1',
    description='deep learning utility',
    author='Yu Murata',
    author_email='me@kennethreitz.com',
    url='https://github.com/kennethreitz/samplemod',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy'],
)