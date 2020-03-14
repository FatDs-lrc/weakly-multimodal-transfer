from .base_opts import BaseOptions


class TestOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--method', type=str, default='mean', help='How to calculate final test result, [concat, mean]')
        self.isTrain = False
        return parser
