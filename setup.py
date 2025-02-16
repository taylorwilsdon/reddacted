#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='reddacted',
    version='1.0',
    license='MIT',

    description='reddacted',
    long_description='Analyze the content of comments to identify anything that might be likely to reveal PII that you may not want correlated with your anonymous username and perform sentiment analysis on the content of those posts',
    author='Taylor Wilsdon',
    author_email='taylorwilsdon@gmail.com',
    url='https://github.com/taylorwilsdon/reddacted',
    download_url='https://github.com/taylorwilsdon/reddacted/archive/refs/heads/master.zip',

    classifiers=['Development Status :: 1 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Software Development :: Build Tools',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.11',
                 'Environment :: Console',
                 ],
    keywords = ['reddact', 'reddacted', 'reddit', 'llm', 'pii', 'sentiment', 'analysis', 'nlp'],
    platforms=['Any'],

    scripts=[],
    provides=[],
    install_requires=['cliff', 'praw', 'nltk', 'requests', 'six', 'openai', 'rich'],
    namespace_packages=[],
    packages=find_packages(include=['reddacted', 'reddacted.*']),
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'reddacted = reddacted.cli:main'
        ],
        'reddacted.analyze': [
            'listing = reddacted.cli:Listing',
            'user = reddacted.cli:User'
        ],
    },
)
