import sys

from cliff.app import App
from cliff.commandmanager import CommandManager
from cliff.command import Command

from reddit_sentiment.sentiment import Sentiment


class Listing(Command):

    def get_description(self):
        return 'get the sentiment score of a post.'

    def get_parser(self, prog_name):
        parser = super(Listing, self).get_parser(prog_name)
        parser.add_argument('subreddit', help='The subreddit.')
        parser.add_argument('article', help='The id of the article.')
        parser.add_argument('--output-file', '-o',
                            help='Outputs a file with information on each '
                                 'sentence of the post, as well as the final '
                                 'score.')
        parser.add_argument('--enable-auth', '-a', type=bool,
                            help='Enable reddit api authentication by '
                            'using the environment variables '
                            'REDDIT_USERNAME '
                            'REDDIT_PASSWORD '
                            'REDDIT_CLIENT_ID '
                            'REDDIT_CLIENT_SECRET')
        return parser

    def take_action(self, args):
        sent = Sentiment(args.enable_auth)
        sent.get_listing_sentiment(args.subreddit,
                                   args.article,
                                   args.output_file)

        if args.output_file:
            print("Listing Contents Outputted to {output_file}".format(
                output_file=args.output_file))


class User(Command):

    def get_description(self):
        return 'get the sentiment score of a user.'

    def get_parser(self, prog_name):
        parser = super(User, self).get_parser(prog_name)
        parser.add_argument('username', help='The name of the user.')
        parser.add_argument('--output-file', '-o',
                            help='Outputs a file with information on each '
                                 'sentence of the post, as well as the final '
                                 'score.')
        parser.add_argument('--enable-auth', '-a', type=bool,
                            help='Enable reddit api authentication by '
                            'using the environment variables '
                            'REDDIT_USERNAME '
                            'REDDIT_PASSWORD '
                            'REDDIT_CLIENT_ID '
                            'REDDIT_CLIENT_SECRET')
        return parser

    def take_action(self, args):
        sent = Sentiment(args.enable_auth)
        sent.get_user_sentiment(args.username, args.output_file)

        if args.output_file:
            print("Listing Contents Outputted to {output_file}".format(
                 output_file=args.output_file))


class CLI(App):
    def __init__(self):
        super(CLI, self).__init__(
            version=1.0,
            description="Obtains Sentiment Score of various reddit objects.",
            command_manager=CommandManager('reddit.sentiment'),
            deferred_help=True,)


def main(argv=sys.argv[1:]):
    app = CLI()
    return app.run(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
