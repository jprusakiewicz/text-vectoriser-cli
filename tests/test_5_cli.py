import os

from typer.testing import CliRunner
from text_matcher.cli import cli_app

runner = CliRunner()


def test_with_no_additional_params():
    result = runner.invoke(cli_app, ["train"])

    # Assert that the command fails
    assert result.exit_code == 0

    # cleanup
    file_path = 'vectorizer_model.pkl'
    if os.path.exists(file_path):
        os.remove(file_path)


def test_count_vectorizer_with_tfidf_param():
    # Invalid parameter for CountVectorizer
    invalid_params = '{"use_idf": true}'

    result = runner.invoke(cli_app, ["train", "--vectorizer-type", "count", '--vectorizer-params', f'{invalid_params}'])

    # Assert that the command fails
    assert result.exit_code != 0
    assert "Invalid vectorizer parameters" in result.output


def test_many_query_documents_from_file():
    runner.invoke(cli_app, ["train"])
    runner.invoke(cli_app, ["pick-best", "data/queries.csv"])

    result = runner.invoke(cli_app, ["pick-best", "data/queries.csv"])

    # Assert that the command fails
    assert result.exit_code == 0

    # cleanup
    file_path = 'vectorizer_model.pkl'
    if os.path.exists(file_path):
        os.remove(file_path)
