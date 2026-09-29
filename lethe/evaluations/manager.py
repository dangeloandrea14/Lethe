import sys
import traceback

from lethe.core.base import Configurable
from lethe.core.factory_base import get_instance_kvargs
from lethe.evaluations.evaluation import Evaluation
from lethe.evaluations.running import UnlearnRunner
from lethe.utils.config.global_ctx import Global
from lethe.core.unlearner import Unlearner
from lethe.utils.config.local_ctx import Local


class Evaluator(Configurable):

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.__init_measures__()

    def evaluate(self, unlearner: Unlearner, predictor):
        e = Evaluation(unlearner,predictor)
        for measure in self.measures:
            try:
                e = measure.process(e)
            except Exception as err:
                exc_type, exc_value, exc_tb = sys.exc_info()
                traceback_details = traceback.format_exception(exc_type, exc_value, exc_tb)
                print("".join(traceback_details))

        return e

    def __init_measures__(self):
        self.measures = []
        for measure in self.params['measures']:
            current = Local(measure)
            self.measures.append( self.global_ctx.factory.get_object(current) )

        # the first metric has to be one that calls the unlearn() method of the unlearner
        if not isinstance(self.measures[0], UnlearnRunner):
            config = {"class": "lethe.evaluations.running.UnlearnRunner", "parameters":{}}
            current = Local(config)
            self.measures.insert(0, self.global_ctx.factory.get_object(current))

        assert isinstance(self.measures[0], UnlearnRunner)

