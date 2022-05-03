import unittest

from alpa.testing import PipelineBasicTest
from alpa.global_env import global_config
from alpa.util import count_communication_primitives

as_option = global_config.default_autosharding_option


class PipelineReduceScatterTest(PipelineBasicTest):

    def test_mlp_grad_acc_friendly(self):
        as_option.force_data_parallel = True
        as_option.prefer_reduce_scatter = True
        hlo_text = self.run_mlp(do_numerical_test=True)

        # Check number of communication primitives
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[0],
                                           ignore_scalar_all_reduce=True))
        assert n_total == 0

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[1],
                                           ignore_scalar_all_reduce=True))
        assert n_total == 0

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[2],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_reduce == 1

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[3],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_reduce == 1

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[4],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_gather == 1

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[5],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_gather == 1

    def test_bert_grad_acc_friendly(self):
        as_option.force_data_parallel = True
        as_option.prefer_reduce_scatter = True
        hlo_text = self.run_n_layer_bert(n_layers=2, do_numerical_test=True)

        # Check numbers of communication primitives
        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[0],
                                           ignore_scalar_all_reduce=True))
        assert n_total == 0

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[1],
                                           ignore_scalar_all_reduce=True))
        assert n_total == 0

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[2],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_reduce == 1

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[3],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_reduce == 1

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[4],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_gather == 1

        n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
            count_communication_primitives(hlo_text[5],
                                           ignore_scalar_all_reduce=True))
        assert n_total == n_all_gather == 1


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineReduceScatterTest('test_mlp_grad_acc_friendly'))
    suite.addTest(PipelineReduceScatterTest('test_bert_grad_acc_friendly'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
