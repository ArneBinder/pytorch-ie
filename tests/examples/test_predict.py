import pytest

from tests.helpers.run_command import run_command


@pytest.mark.slow
def test_predict_ner_span_classification():
    command = ["examples/predict/ner_span_classification.py"]
    run_command(command)


@pytest.mark.slow
def test_predict_re_generative():
    command = ["examples/predict/re_generative.py"]
    run_command(command)


@pytest.mark.slow
def test_predict_re_text_classification():
    command = ["examples/predict/re_text_classification.py"]
    run_command(command)
