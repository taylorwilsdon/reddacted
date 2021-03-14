import logging
import sys
import warnings

from cliff.app import App
from cliff.commandmanager import CommandManager
from cliff.command import Command

from miner import Miner


class Listing(Command):

    def get_description(self):
        return 'print the sentiment score of a post.'

    def get_parser(self, prog_name):
        parser = super(Listing, self).get_parser(prog_name)
        parser.add_argument('subreddit', help='The subreddit.')
        parser.add_argument('article', help='The id of the article.')
        parser.add_argument('--output-file', '-o',
                            help='Outputs a file with information on each '
                                 'sentence of the post, as well as the final '
                                 'score.')
        return parser

    def take_action(self, args):
        miner = Miner()
        miner.get_listing_sentiment(args.subreddit,
                                    args.article,
                                    args.output_file)

        if args.output_file:
            print("Listing Contents Outputted to {output_file}".format(
                output_file=args.output_file))


class User(Command):

    def get_description(self):
        return 'print the sentiment score of a user.'

    def get_parser(self, prog_name):
        parser = super(User, self).get_parser(prog_name)
        parser.add_argument('username', help='The name of the user.')
        parser.add_argument('--output-file', '-o',
                            help='Outputs a file with information on each '
                                 'sentence of the post, as well as the final '
                                 'score.')
        return parser

    def take_action(self, args):
        miner = Miner()
        miner.get_user_sentiment(args.username, args.output_file)

        if args.output_file:
            print("Listing Contents Outputted to {output_file}".format(
                 output_file=args.output_file))


class CLI(App):

    def __init__(self):
        super(CLI, self).__init__(
            description='Reddit Sentiment Analyzer',
            version='1.0',
            command_manager=CommandManager('reddit.analyze'),
            deferred_help=True,)


def main(argv=sys.argv[1:]):
    myapp = CLI()
    return myapp.run(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
