import requests
import unittest
import json

from unittest.mock import MagicMock

from reddit_sentiment.api import scraper


class ScraperPostiveTestCases(unittest.TestCase):

    def setUp(self):
        super(ScraperPostiveTestCases, self).setUp()
        self.maxDiff = None

    def tearDown(self):
        super(ScraperPostiveTestCases, self).tearDown()

    def test_parse_user(self):
        # Mocking the Request for obtaining json from Reddit
        user_data = ""
        with open("tests/templates/reddit/user.json") as user_file:
            user_data = json.loads(user_file.read())
        valid_user_response = MagicMock(name='mock_response')
        valid_user_response.json = MagicMock(return_value=user_data)
        valid_user_response.status_code = 200
        requests.get = MagicMock(return_value=valid_user_response)
        
        expected = ['Sure is! Appended it to my tweet.',
                    'Anyone not treating the animals properly, should simply be banned after a warning.',
                    'Is it safe to shower using head and shoulders once per day?',
                    'Its crazy how this flows so well in my mind',
                    "Cuban's love the name fifi for girl dogs. Source: Me a Cuban American.",
                    'You can simply follow the deployment guide for Ingress-NGINX, if that is the controller you are wanting to use. See [https://github.com/kubernetes/ingress-nginx/blob/master/docs/deploy/index.md](https://github.com/kubernetes/ingress-nginx/blob/master/docs/deploy/index.md)  When you create your ingress resource, you can specify the host as [www.example.com](https://www.example.com) and in your /etc/hosts you can put that URL as the clusterIP. Then send a curl using the \\`Host\\` header to verify. I made a tutorial a while back, using minikube, but the example should still work on an AWS cluster. See [https://medium.com/@awkwardferny/getting-started-with-kubernetes-ingress-nginx-on-minikube-d75e58f52b6c](https://medium.com/@awkwardferny/getting-started-with-kubernetes-ingress-nginx-on-minikube-d75e58f52b6c) Also if you still have questions, you can always post on [http://slack.k8s.io/](http://slack.k8s.io/) on the #ingress-nginx channel.',
                    'Hey u/Jokkamo Seems like the syntax is off in the template. I created a blog about templating : [https://medium.com/@awkwardferny/golang-templating-made-easy-4d69d663c558](https://medium.com/@awkwardferny/golang-templating-made-easy-4d69d663c558). Hope it helps you!! You could also create a template function to examine currentTitle. ',
                    "I guess that's a good one to add lol.",
                    'RaunchyRaccoon that looks a lot like Miami Springs!',
                    "If you can't find water anywhere, I thought of a solution. Simply buy some cheap sodas/tea and drain the soda away and fill it up with tap-water! Will at least keep you with some water.",
                    'You ever been in a storm? https://www.youtube.com/watch?v=Pr7Y0kZ67o0',
                    'Officer Joseph.']

        sc = scraper.Scraper()
        result = sc.parse_user("awkwardferny")

        self.assertEqual(expected, result)

    def test_parse_listing(self):
        # Mocking the Request for obtaining json from Reddit
        article_data = ""
        with open("tests/templates/reddit/article.json") as article_file:
            article_data = json.loads(article_file.read())
        valid_article_response = MagicMock(name='mock_response')
        valid_article_response.json = MagicMock(return_value=article_data)
        valid_article_response.status_code = 200
        requests.get = MagicMock(return_value=valid_article_response)
        
        expected = ['Looks sick!',
                    '#DOGE HOLD IT',
                    '10/10 very art such picasso wow',
                    'Much drawing, very sketch. Ps I gave you the silver award.']

        sc = scraper.Scraper()
        result = sc.parse_listing("doge", "l7zp94")

        self.assertEqual(expected, result)