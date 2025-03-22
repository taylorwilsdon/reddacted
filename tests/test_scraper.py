import requests
import unittest
import json

from unittest.mock import MagicMock

from reddacted.api import scraper


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
        valid_user_response = MagicMock(name="mock_response")
        valid_user_response.json = MagicMock(return_value=user_data)
        valid_user_response.status_code = 200
        requests.get = MagicMock(return_value=valid_user_response)

        expected = [
            {
                "text": "Sure is! Appended it to my tweet.",
                "upvotes": 1,
                "downvotes": 0,
                "id": "fnq55o3",
                "permalink": "/r/WhitePeopleTwitter/comments/g35yge/the_battle_cry_of_a_generation/fnq55o3/",
            },
            {
                "text": "Anyone not treating the animals properly, should simply be banned after a warning.",
                "upvotes": 2,
                "downvotes": 0,
                "id": "eyvcagh",
                "permalink": "/r/AmItheAsshole/comments/cyt2nl/aita_for_allowing_two_teenagers_to_be_spit_on_by/eyvcagh/",
            },
            {
                "text": "Is it safe to shower using head and shoulders once per day?",
                "upvotes": 4,
                "downvotes": 0,
                "id": "eyvbx4k",
                "permalink": "/r/science/comments/cyx8s4/teen_went_blind_after_eating_only_pringles_fries/eyvbx4k/",
            },
            {
                "text": "Its crazy how this flows so well in my mind",
                "upvotes": 2,
                "downvotes": 0,
                "id": "ex2ovgj",
                "permalink": "/r/memes/comments/cr86z6/dr_phil_review_this/ex2ovgj/",
            },
            {
                "text": "Cuban's love the name fifi for girl dogs. Source: Me a Cuban American.",
                "upvotes": 1,
                "downvotes": 0,
                "id": "ewx8paz",
                "permalink": "/r/AskReddit/comments/cqdjg6/nonamericans_does_your_culture_have_oldfashioned/ewx8paz/",
            },
            {
                "text": "You can simply follow the deployment guide for Ingress-NGINX, if that is the controller you are wanting to use. See [https://github.com/kubernetes/ingress-nginx/blob/master/docs/deploy/index.md](https://github.com/kubernetes/ingress-nginx/blob/master/docs/deploy/index.md)  When you create your ingress resource, you can specify the host as [www.example.com](https://www.example.com) and in your /etc/hosts you can put that URL as the clusterIP. Then send a curl using the \\`Host\\` header to verify. I made a tutorial a while back, using minikube, but the example should still work on an AWS cluster. See [https://medium.com/@awkwardferny/getting-started-with-kubernetes-ingress-nginx-on-minikube-d75e58f52b6c](https://medium.com/@awkwardferny/getting-started-with-kubernetes-ingress-nginx-on-minikube-d75e58f52b6c) Also if you still have questions, you can always post on [http://slack.k8s.io/](http://slack.k8s.io/) on the #ingress-nginx channel.",
                "upvotes": 2,
                "downvotes": 0,
                "id": "ehsepvh",
                "permalink": "/r/kubernetes/comments/awvv0h/how_to_create_an_ingress_controller_on_cluster/ehsepvh/",
            },
            {
                "text": "Hey u/Jokkamo Seems like the syntax is off in the template. I created a blog about templating : [https://medium.com/@awkwardferny/golang-templating-made-easy-4d69d663c558](https://medium.com/@awkwardferny/golang-templating-made-easy-4d69d663c558). Hope it helps you!! You could also create a template function to examine currentTitle. ",
                "upvotes": 1,
                "downvotes": 0,
                "id": "ee4r5v2",
                "permalink": "/r/golang/comments/afxhvk/how_can_i_check_where_a_variable_defined_in_html/ee4r5v2/",
            },
            {
                "text": "I guess that's a good one to add lol.",
                "upvotes": 2,
                "downvotes": 0,
                "id": "e5ezchx",
                "permalink": "/r/programming/comments/9d1fh5/bad_software_development_patterns_and_how_to_fix/e5ezchx/",
            },
            {
                "text": "RaunchyRaccoon that looks a lot like Miami Springs!",
                "upvotes": 1,
                "downvotes": 0,
                "id": "dmvmihx",
                "permalink": "/r/HumansBeingBros/comments/6zgfvk/our_neighborhood_got_battered_by_irma_many/dmvmihx/",
            },
            {
                "text": "If you can't find water anywhere, I thought of a solution. Simply buy some cheap sodas/tea and drain the soda away and fill it up with tap-water! Will at least keep you with some water.",
                "upvotes": 2,
                "downvotes": 0,
                "id": "dmnmuve",
                "permalink": "/r/Miami/comments/6ydvec/hurricane_irma_megathread_2_97/dmnmuve/",
            },
            {
                "text": "You ever been in a storm? https://www.youtube.com/watch?v=Pr7Y0kZ67o0",
                "upvotes": 1,
                "downvotes": 0,
                "id": "dld5va0",
                "permalink": "/r/worldnews/comments/6sfvxd/trump_if_north_korea_escalates_nuclear_threat/dld5va0/",
            },
            {
                "text": "Officer Joseph.",
                "upvotes": 1,
                "downvotes": 0,
                "id": "dggdqs0",
                "permalink": "/r/funny/comments/6664cj/look_whos_taking_the_picture/dggdqs0/",
            },
        ]

        sc = scraper.Scraper()
        result = sc.parse_user("awkwardferny")

        self.assertEqual(expected, result)

    def test_parse_listing(self):
        # Mocking the Request for obtaining json from Reddit
        article_data = ""
        with open("tests/templates/reddit/article.json") as article_file:
            article_data = json.loads(article_file.read())
        valid_article_response = MagicMock(name="mock_response")
        valid_article_response.json = MagicMock(return_value=article_data)
        valid_article_response.status_code = 200
        requests.get = MagicMock(return_value=valid_article_response)

        expected = [
            {
                "text": "Looks sick!",
                "upvotes": 4,
                "downvotes": 0,
                "id": "glai61b",
                "permalink": "/r/doge/comments/l7zp94/i_drew_this_doge_in_2013_during_my_first_years_of/glai61b/",
            },
            {
                "text": "#DOGE HOLD IT",
                "upvotes": 1,
                "downvotes": 0,
                "id": "glc7p50",
                "permalink": "/r/doge/comments/l7zp94/i_drew_this_doge_in_2013_during_my_first_years_of/glc7p50/",
            },
            {
                "text": "10/10 very art such picasso wow",
                "upvotes": 1,
                "downvotes": 0,
                "id": "gladezi",
                "permalink": "/r/doge/comments/l7zp94/i_drew_this_doge_in_2013_during_my_first_years_of/gladezi/",
            },
            {
                "text": "Much drawing, very sketch. Ps I gave you the silver award.",
                "upvotes": 2,
                "downvotes": 0,
                "id": "gsdozky",
                "permalink": "/r/doge/comments/l7zp94/i_drew_this_doge_in_2013_during_my_first_years_of/gsdozky/",
            },
        ]

        sc = scraper.Scraper()
        result = sc.parse_listing("doge", "l7zp94")

        self.assertEqual(expected, result)
