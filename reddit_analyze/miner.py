#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import warnings

from api.scraper import Scraper

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Miner():
    """Performs the sentiment analysis on a given set of Reddit Objects."""

    def __init__(self):
        self.api = Scraper()
        self.score = 0
        self.sentiment = "¯\_(ツ)_/¯"

    def get_user_sentiment(self, username, output_file=None):
        """Obtains the sentiment for a user's comments.

        :param username: name of user to search
        :param output_file (optional): file to output relevant data.
        """
        self.comments = self.api.parse_user(username)
        self.score = self._analyze(self.comments)
        self.sentiment = self._get_sentiment(self.score)

        # TODO(diazjf): append sentiment calculation to end of each sentence
        # for more detailed analysis
        if output_file:
            target = open(output_file, 'w+')
            for comment in self.comments:
                target.write(str(comment) + "\n")
            target.write("\nScore: {score} \n".format(score=self.score))
            target.write("Sentiment: {sentiment}".format(
                sentiment=self.sentiment))
        else:
            for comment in self.comments:
                print comment

    def get_listing_sentiment(self, subreddit, article, output_file=None):
        """Obtains the sentiment for a listing's comments.

        :param subreddit: a subreddit
        :param article: an article associated with the subreddit
        :param output_file (optional): file to output relevant data.
        """
        self.comments = self.api.parse_listing(subreddit, article)
        self.score = self._analyze(self.comments)
        self.sentiment = self._get_sentiment(self.score)

        # TODO(diazjf): append sentiment calculation to end of each sentence
        # for more detailed analysis
        if output_file:
            target = open(output_file, 'w+')
            for comment in self.comments:
                target.write(str(comment) + "\n")
            target.write("\nScore: {score} \n".format(score=self.score))
            target.write("Sentiment: {sentiment}".format(
                sentiment=self.sentiment))
        else:
            for comment in self.comments:
                print comment

    def _analyze(self, comments):
        """Obtains the sentiment for a user's comments.

        :param comments: comments to perform analysis on.

        :return: combined sentiment score of all sentences.
        """
        sentiment_analyzer = SentimentIntensityAnalyzer()
        final_score = 0

        cleanup_regex = re.compile('<.*?>')

        for comment in comments:
            comment = re.sub(cleanup_regex, '', str(comment))

            # provides compound, negative, neutral, and positive scores
            all_scores = sentiment_analyzer.polarity_scores(comment)
            score = all_scores['compound']
            final_score = final_score + score

        # return the average of score of all comments.
        return round(final_score/len(comments), 4)

    def _get_sentiment(self, score):
        """Obtains the sentiment using a sentiment score.

        :param score: the sentiment score.

        :return: sentiment from score.
        """
        if score == 0:
            return "¯\_(ツ)_/¯"
        elif score > 0:
            return "ʘ‿ʘ"
        else:
            return "ಠ_ಠ"
