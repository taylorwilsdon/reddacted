#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='reddit-sentiment',
    version='1.0',
    license='MIT',

    description='Reddit Sentiment Analyzer',
    long_description='Obtains Sentiment Score of various reddit objects.',

    author='Fernando Diaz',
    author_email='awkwardferny@gmail.com',
    url='https://github.com/fishtoadsoft/reddit-sentiment-analyzer',
    download_url='https://github.com/fishtoadsoft/reddit-sentiment-analyzer/archive/refs/tags/v1.0.0.tar.gz',

    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Software Development :: Build Tools',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.10',
                 'Environment :: Console',
                 ],
    keywords = ['reddit', 'sentiment', 'analysis', 'nlp'],
    platforms=['Any'],

    scripts=[],
    provides=[],
    install_requires=['cliff', 'praw', 'nltk', 'requests', 'six'],
    namespace_packages=[],
    packages=find_packages(),
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'reddit-sentiment = reddit_sentiment.cli:main'
        ],
        'reddit.sentiment': [
            'listing = reddit_sentiment.cli:Listing',
            'user = reddit_sentiment.cli:User'
        ],
    },
)