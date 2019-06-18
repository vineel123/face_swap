import argparse
import os 

class FullPaths(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Expand user- and relative-paths """
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            vals = [os.path.abspath(os.path.expanduser(val)) for val in values]
            print("yes")
        else:
            vals = os.path.abspath(os.path.expanduser(values))
        setattr(namespace, self.dest, vals)



    

class MulTrainArgs():
    """ Class to parse the command line arguments for training """
    def __init__(self, subparser, command,
                 description="default", subparsers=None):

        self.global_arguments = []
        self.argument_list = self.get_argument_list()
        self.optional_arguments = []
        if not subparser:
            return

        self.parser = self.create_parser(subparser, command, description)

        self.add_arguments()

        #script = ScriptExecutor(command, subparsers)
        #self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ("-A", "--input-A"),
        	                  "nargs":"+",
                              "action": FullPaths,
                              "dest": "input_a",
                              "required": True,
                              "help": "Input directory. A directory containing training images "
                                      "for face A."})
        return argument_list

    @staticmethod
    def create_parser(subparser, command, description):
        """ Create the parser for the selected command """
        parser = subparser.add_parser(
            command,
            help=description,
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground",
            )
        return parser

    def add_arguments(self):
        """ Parse the arguments passed in from argparse """
        options = self.global_arguments + self.argument_list + self.optional_arguments
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key]
                      for key in option.keys() if key != "opts"}
            self.parser.add_argument(*args, **kwargs)


PARSER = argparse.ArgumentParser()
SUBPARSER = PARSER.add_subparsers()
MULTRAIN = MulTrainArgs(SUBPARSER , "mul_train" ,"This command trains a model for multiple faces")
ARGUMENTS = PARSER.parse_args()
print(ARGUMENTS)