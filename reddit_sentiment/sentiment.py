#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from reddit_sentiment.api.scraper import Scraper
from reddit_sentiment.api.reddit import Reddit
from reddit_sentiment.pii_detector import PIIDetector
from nltk.sentiment.vader import SentimentIntensityAnalyzer


@dataclass
class AnalysisResult:
    """Holds the results of both sentiment and PII analysis"""
    sentiment_score: float
    sentiment_emoji: str
    pii_risk_score: float
    pii_matches: List[Any]
    text: str

happy_sentiment = "😁"
sad_sentiment = "😕"
neutral_sentiment = "😐"


class Sentiment():
    """Performs the sentiment analysis on a given set of Reddit Objects."""

    def __init__(self, auth_enabled=False):
        self.api = Scraper()
        self.score = 0
        self.sentiment = neutral_sentiment
        self.headers = {'User-agent': "Reddit Sentiment Analyzer"}
        self.authEnable = False
        self.pii_detector = PIIDetector()
        
        if auth_enabled:
            self.api = Reddit()

    def get_user_sentiment(self, username, output_file=None):
        """Obtains the sentiment for a user's comments.

        :param username: name of user to search
        :param output_file (optional): file to output relevant data.
        """
        comments = self.api.parse_user(username, headers=self.headers)
        self.score = self._analyze(comments)
        self.sentiment = self._get_sentiment(self.score)

        user_id = "/user/{username}".format(username=username)

        if output_file:
            self._generate_output_file(output_file, comments, user_id)
        else:
             self._print_comments(comments, user_id)

    def get_listing_sentiment(self, subreddit, article, output_file=None):
        """Obtains the sentiment for a listing's comments.

        :param subreddit: a subreddit
        :param article: an article associated with the subreddit
        :param output_file (optional): file to output relevant data.
        """
        comments = self.api.parse_listing(subreddit,
                                          article,
                                          headers=self.headers)
        self.score = self._analyze(comments)
        self.sentiment = self._get_sentiment(self.score)

        article_id = "/r/{subreddit}/comments/{article}".format(subreddit=subreddit,
                                                                article=article)

        if output_file:
            self._generate_output_file(output_file, comments, article_id)
        else:
             self._print_comments(comments, article_id)

    def _analyze(self, comments):
        """Analyzes comments for both sentiment and PII content.

        :param comments: comments to perform analysis on.
        :return: tuple of (sentiment_score, list of AnalysisResult objects)
        """
        sentiment_analyzer = SentimentIntensityAnalyzer()
        final_score = 0
        results = []

        cleanup_regex = re.compile('<.*?>')

        for comment in comments:
            clean_comment = re.sub(cleanup_regex, '', str(comment))
            
            # Sentiment analysis
            all_scores = sentiment_analyzer.polarity_scores(clean_comment)
            score = all_scores['compound']
            final_score += score

            # PII analysis
            pii_risk_score, pii_matches = self.pii_detector.get_pii_risk_score(clean_comment)
            
            results.append(AnalysisResult(
                sentiment_score=score,
                sentiment_emoji=self._get_sentiment(score),
                pii_risk_score=pii_risk_score,
                pii_matches=pii_matches,
                text=clean_comment
            ))

        try:
            rounded_final = round(final_score/len(comments), 4)
            return rounded_final, results
        except ZeroDivisionError:
            logging.error("No comments found")
            return 0.0, []

    def _get_sentiment(self, score):
        """Obtains the sentiment using a sentiment score.

        :param score: the sentiment score.
        :return: sentiment from score.
        """
        if score == 0:
            return neutral_sentiment
        elif score > 0:
            return happy_sentiment
        else:
            return sad_sentiment

    def _generate_output_file(self, filename, comments, url):
        """Outputs a file containing a detailed sentiment and PII analysis per
        sentence.

        :param: filename: the name of the file to create and edit
        :param: comments: the parsed contents to analyze.
        :param: url: the url being parsed.
        """
        with open(filename, 'w+') as target:
            target.write(f"Analysis for '{url}'\n")
            target.write(f"Overall Sentiment Score: {self.score}\n")
            target.write(f"Overall Sentiment: {self.sentiment}\n\n")

            comment_count = 1
            for comment in comments:
                score, results = self._analyze([comment])
                for result in results:
                    target.write(f"Comment {comment_count}:\n")
                    target.write(f"Text: {result.text}\n")
                    target.write(f"Sentiment Score: {result.sentiment_score}\n")
                    target.write(f"Sentiment: {result.sentiment_emoji}\n")
                    target.write(f"PII Risk Score: {result.pii_risk_score:.2f}\n")
                    
                    if result.pii_matches:
                        target.write("PII Detected:\n")
                        for pii in result.pii_matches:
                            target.write(f"  - Type: {pii.type}\n")
                            target.write(f"    Confidence: {pii.confidence:.2f}\n")
                    target.write("\n")
                    
                comment_count += 1


    def _print_comments(self, comments, url):
        """Prints out analysis of user comments.

        :param: comments: the parsed contents to analyze.
        :param: url: the url being parsed.
        """
        print(f"Analysis for '{url}'")
        print(f"Overall Sentiment Score: {self.score}")
        print(f"Overall Sentiment: {self.sentiment}\n")

        comment_count = 1
        for comment in comments:
            score, results = self._analyze([comment])
            for result in results:
                print(f"Comment {comment_count}:")
                print(f"Text: {result.text}")
                print(f"Sentiment Score: {result.sentiment_score}")
                print(f"Sentiment: {result.sentiment_emoji}")
                print(f"PII Risk Score: {result.pii_risk_score:.2f}")
                
                if result.pii_matches:
                    print("PII Detected:")
                    for pii in result.pii_matches:
                        print(f"  - Type: {pii.type}")
                        print(f"    Confidence: {pii.confidence:.2f}")
                print()
                
            comment_count += 1
  
