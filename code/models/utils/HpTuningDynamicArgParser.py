from utils.utils import read_yaml
from pydoc import locate

from utils.utils import read_yaml


class HpTuningDynamicArgParser:

    def __init__(self, path_to_hp_tuning_parser_conf, parser, classifier_name):
        self.path_to_hp_tuning_parser_conf = path_to_hp_tuning_parser_conf
        self.parser = parser
        self.hp_tuning_parser_conf = read_yaml(self.path_to_hp_tuning_parser_conf)
        classifier_list = self.hp_tuning_parser_conf['hp_tuning_parser_conf']
        self.classifier_params = [classifier_spec for classifier_spec in classifier_list if classifier_spec['classifier_name'] == classifier_name][0]

    def build_dynamic_parser_arguments(self):
        for param in self.classifier_params['arg_params']:
            self.parser.add_argument(param['name'], type=locate(param['type']), default=param['default'])
        return self.parser

    def get_classifier_args_dict(self, args):
        args_dict = vars(args)
        classifier_args = dict()
        for param in self.classifier_params['arg_params']:
            param_name = param['name'].replace("--", "")
            # None parameters are not added to classifier_args: this avoids bugs when passing null hyperparameters to fit()
            if args_dict[param_name] != None :
                classifier_args[param_name] = args_dict[param_name]
        return classifier_args


