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

        if output_file:
            self._generate_output_file(output_file, self.comments)
        else:
            # TODO: Move this to common function
            for comment in self.comments:
                print comment
            print "\n"
            print "Score: {score}".format(score=self.score)
            print "Sentiment: {sentiment}".format(sentiment=self.sentiment)

    def get_listing_sentiment(self, subreddit, article, output_file=None):
        """Obtains the sentiment for a listing's comments.

        :param subreddit: a subreddit
        :param article: an article associated with the subreddit
        :param output_file (optional): file to output relevant data.
        """
        self.comments = self.api.parse_listing(subreddit, article)
        self.score = self._analyze(self.comments)
        self.sentiment = self._get_sentiment(self.score)

        if output_file:
            self._generate_output_file(output_file, comments)
        else:
            # TODO: Move this to common function
            for comment in self.comments:
                print comment
            print "\n"
            print "Score: {score}".format(score=self.score)
            print "Sentiment: {sentiment}".format(sentiment=self.sentiment)

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

    def _generate_output_file(self, filename, comments):
        """Outputs a file containing a detailed sentiment analysis per
        sentence.

        :param: filename: the name of the file to create and edit
        :param: comments: the parsed contents to analyze.
        """
        target = open(filename, 'w+')

        for comment in comments:
            score = self._analyze(comment)
            sentiment = self._get_sentiment(score)

            # TODO: Create a new line if the sentence takes up over 80 chars
            target.write("Sentence: {sentence}\nScore: {score}, Sentiment: "
                         "{sentiment}\n".format(sentence=str(comment),
                                                score=score,
                                                sentiment=sentiment))

        target.write("\nFinal Score: {score} \n".format(score=self.score))
        target.write("Final Sentiment: {sentiment}".format(
            sentiment=self.sentiment))
