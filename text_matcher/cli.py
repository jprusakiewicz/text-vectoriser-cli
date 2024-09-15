from typing import Optional
from urllib.parse import urlparse

import typer
from pydantic import ValidationError

from text_matcher.core import train_and_save_vectorizer, load_vectorizer_and_pick_best, \
    load_vectorizer_and_pick_best_for_all, load_data
from text_matcher.vectorizer_config import build_vectorizer_config

cli_app = typer.Typer()


@cli_app.command()
def train(
        vectorizer_type: str = typer.Option("count",
                                            help="Vectorizer type: count, tfidf, hashing"),
        train_file_path: str = typer.Option("data/train.csv",
                                            help="Wikipedia URLs csv file path"),
        output_model_path: str = typer.Option("vectorizer_model.pkl",
                                              help="Path to save the model"),
        vectorizer_params: Optional[str] = "{}"

):
    try:
        vectorizer_config = build_vectorizer_config(vectorizer_type, vectorizer_params)
    except ValidationError:
        typer.echo(f"Invalid vectorizer parameters {vectorizer_params} for {vectorizer_type} vectorizer.")
        raise typer.Exit(1)
    train_and_save_vectorizer(output_model_path, train_file_path, vectorizer_config)
    typer.echo(f"Model {vectorizer_type} saved as {output_model_path}.")


def is_file(path) -> bool:
    if ".csv" in path:
        return True
    return False


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@cli_app.command()
def pick_best(
        query: str = typer.Argument(...,
                                    help="A single URL or a path to a csv file containing URLs to find the best match for"),
        documents_path: str = typer.Option("data/test.csv",
                                           help="Path to the CSV file with document to be matched URLs"),
        vectorizer_path: str = typer.Option("vectorizer_model.pkl",
                                            help="Path to the saved vectorizer model"),
        distance_metric: str = typer.Option("cosine",
                                            help="Distance metric: cosine, euclidean, manhattan")
):
    if is_file(query):
        query_urls = load_data(str(query))
        best_matches = load_vectorizer_and_pick_best_for_all(distance_metric, query_urls, documents_path,
                                                             vectorizer_path)
        for query_url, best_match in best_matches.items():
            typer.echo(f"Best match for {query_url} is: {best_match}")
    elif is_valid_url(query):
        best_match = load_vectorizer_and_pick_best(distance_metric, str(query), documents_path,
                                                   vectorizer_path)
        typer.echo(f"Best match: {best_match}")
    else:
        typer.echo(f"provided query_url_or_file_path is not a valid URL or file path: {query}")
        raise typer.Exit(1)


if __name__ == "__main__":
    cli_app()
