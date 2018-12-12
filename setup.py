from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='my_deep_learning',
    version='0.0.1',
    description='deep learning utility',
    long_description=readme,
    author='Yu Murata',
    author_email='me@kennethreitz.com',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'tensorflow'
        ],
)