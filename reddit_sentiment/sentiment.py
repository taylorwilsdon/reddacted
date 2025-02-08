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
    llm_risk_score: float = 0.0
    llm_findings: Dict[str, Any] = None

happy_sentiment = "😁"
sad_sentiment = "😕"
neutral_sentiment = "😐"


class Sentiment():
    """Performs the sentiment analysis on a given set of Reddit Objects."""

    def __init__(self, auth_enabled=False, pii_enabled=True, llm_config=None, pii_only=False):
        self.api = Scraper()
        self.score = 0
        self.sentiment = neutral_sentiment
        self.headers = {'User-agent': "Reddit Sentiment Analyzer"}
        self.authEnable = False
        self.pii_enabled = pii_enabled
        self.pii_detector = PIIDetector() if pii_enabled else None
        
        # Initialize LLM detector if config provided
        self.llm_detector = None
        if llm_config and pii_enabled:
            from reddit_sentiment.llm_detector import LLMDetector
            self.llm_detector = LLMDetector(
                api_key=llm_config.get('api_key'),
                api_base=llm_config.get('api_base'),
                model=llm_config.get('model', 'gpt-3.5-turbo')
            )
        
        if auth_enabled:
            self.api = Reddit()

    def get_user_sentiment(self, username, output_file=None):
        """Obtains the sentiment for a user's comments.

        :param username: name of user to search
        :param output_file (optional): file to output relevant data.
        """
        comments = self.api.parse_user(username, headers=self.headers)
        self.score, self.results = self._analyze(comments)
        self.sentiment = self._get_sentiment(self.score)

        user_id = f"/user/{username}"

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
        self.score, self.results = self._analyze(comments)
        self.sentiment = self._get_sentiment(self.score)

        article_id = f"/r/{subreddit}/comments/{article}"

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
            pii_risk_score, pii_matches = 0.0, []
            llm_risk_score, llm_findings = 0.0, None
            
            if self.pii_enabled:
                pii_risk_score, pii_matches = self.pii_detector.get_pii_risk_score(clean_comment)
                
                # LLM analysis if enabled
                if self.llm_detector:
                    llm_risk_score, llm_findings = self.llm_detector.analyze_text(clean_comment)
            
            results.append(AnalysisResult(
                sentiment_score=score,
                sentiment_emoji=self._get_sentiment(score),
                pii_risk_score=max(pii_risk_score, llm_risk_score),  # Use highest risk score
                pii_matches=pii_matches,
                text=clean_comment,
                llm_risk_score=llm_risk_score,
                llm_findings=llm_findings
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

            def should_show_result(result):
                if not hasattr(self, 'pii_only') or not self.pii_only:
                    return True
                return result.pii_risk_score >= 1.0

            comment_count = 1
            for comment in comments:
                score, results = self._analyze([comment])
                filtered_results = [r for r in results if should_show_result(r)]
                
                if not filtered_results and hasattr(self, 'pii_only') and self.pii_only:
                    continue
                    
                for result in filtered_results:
                    target.write(f"Comment {comment_count}:\n")
                    target.write(f"Text: {result.text}\n")
                    target.write(f"Sentiment Score: {result.sentiment_score}\n")
                    target.write(f"Sentiment: {result.sentiment_emoji}\n")
                    target.write(f"PII Risk Score: {result.pii_risk_score:.2f}\n")
                    
                    if result.pii_matches:
                        target.write("Pattern-based PII Detected:\n")
                        for pii in result.pii_matches:
                            target.write(f"  - Type: {pii.type}\n")
                            target.write(f"    Confidence: {pii.confidence:.2f}\n")
                    
                    if result.llm_findings:
                        target.write("\nAI Privacy Analysis:\n")
                        target.write(f"  Risk Score: {result.llm_risk_score:.2f}\n")
                        if isinstance(result.llm_findings, dict):
                            if result.llm_findings.get('has_pii'):
                                target.write("  PII Detected: Yes\n")
                            if result.llm_findings.get('details'):
                                target.write("  Findings:\n")
                                for detail in result.llm_findings['details']:
                                    target.write(f"    - {detail}\n")
                            if result.llm_findings.get('reasoning'):
                                target.write(f"\n  Reasoning:\n    {result.llm_findings['reasoning']}\n")
                            if result.llm_findings.get('risk_factors'):
                                target.write("\n  Risk Factors:\n")
                                for factor in result.llm_findings['risk_factors']:
                                    target.write(f"    - {factor}\n")
                            if result.llm_findings.get('recommendations'):
                                target.write("\n  Recommendations:\n")
                                for rec in result.llm_findings['recommendations']:
                                    target.write(f"    - {rec}\n")
                    target.write("\n")
                    
                comment_count += 1


    def _print_comments(self, comments, url):
        """Prints out analysis of user comments.

        :param: comments: the parsed contents to analyze.
        :param: url: the url being parsed.
        """
        def should_show_result(result):
            if not hasattr(self, 'pii_only') or not self.pii_only:
                return True
            return result.pii_risk_score >= 1.0
        print(f"Analysis for '{url}'")
        print(f"Overall Sentiment Score: {self.score}")
        print(f"Overall Sentiment: {self.sentiment}\n")

        # Filter results if pii_only is enabled
        filtered_results = [r for r in self.results if should_show_result(r)]
        
        if hasattr(self, 'pii_only') and self.pii_only and not filtered_results:
            print("No comments with high PII risk found.")
            return

        for i, result in enumerate(filtered_results, 1):
            print(f"Comment {i}:")
            print(f"Text: {result.text}")
            print(f"Sentiment Score: {result.sentiment_score}")
            print(f"Sentiment: {result.sentiment_emoji}")
            print(f"PII Risk Score: {result.pii_risk_score:.2f}")
            
            if result.pii_matches:
                print("\nPattern-based PII Detected:")
                for pii in result.pii_matches:
                    print(f"  - Type: {pii.type}")
                    print(f"    Confidence: {pii.confidence:.2f}")
            
            if result.llm_findings:
                print("\nAI Privacy Analysis:")
                print(f"  Risk Score: {result.llm_risk_score:.2f}")
                if result.llm_findings.get('has_pii'):
                    print("  PII Detected: Yes")
                if result.llm_findings.get('details'):
                    print("  Findings:")
                    for detail in result.llm_findings['details']:
                        print(f"    - {detail}")
                if result.llm_findings.get('reasoning'):
                    print(f"\n  Reasoning:\n    {result.llm_findings['reasoning']}")
                if result.llm_findings.get('risk_factors'):
                    print("\n  Risk Factors:")
                    for factor in result.llm_findings['risk_factors']:
                        print(f"    - {factor}")
                if result.llm_findings.get('recommendations'):
                    print("\n  Recommendations:")
                    for rec in result.llm_findings['recommendations']:
                        print(f"    - {rec}")
            print()
  
