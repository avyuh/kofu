from setuptools import setup, find_packages

setup(
    name='kofu',  # Project name
    version='0.2.1',
    description='An execution framework for i/o heavy task with memory persistence and concurrency',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='avyuh',
    author_email='contact@jhaabhi.com',
    url='https://github.com/avyuh/kofu',  # Your GitHub repo link
    packages=find_packages(),  # Automatically finds all packages in the kofu directory
    include_package_data=True,  # Includes additional files like README and LICENSE
    install_requires=[
        'tqdm'
    ],  # Add external dependencies here, e.g., 'requests'
    extras_require={
        'dev': [
            'pytest>=6.0',  # Adds pytest as a development dependency
        ],
    },
    python_requires='>=3.6',  # Minimum Python version
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
