#!/usr/bin/env python

from setuptools import setup, find_packages

PROJECT = 'reddit-analyze'
VERSION = '2.0'


setup(
    name=PROJECT,
    version=VERSION,

    description='Reddit Sentiment Analyzer',
    long_description='Obtains Sentiment Score of various reddit objects.',

    author='Fernando Diaz',
    author_email='awkwardferny@gmail.com',

    url='https://github.com/diazjf/reddit-sentiment-analyzer',

    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Intended Audience :: Developers',
                 'Environment :: Console',
                 ],

    platforms=['Any'],

    scripts=[],

    provides=[],
    install_requires=['cliff'],

    namespace_packages=[],
    packages=find_packages(),
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'reddit-analyze = reddit_analyze.cli:main'
        ],
        'reddit.analyze': [
            'listing = reddit_analyze.cli:Listing',
            'user = reddit_analyze.cli:User'
        ],
    },

    zip_safe=False,
)
