import typer
from app.core import train_and_save_vectorizer, load_vectorizer_and_pick_best

cli_app = typer.Typer()


@cli_app.command()
def train(
        vectorizer_type: str = typer.Argument("count",
                                              help="Vectorizer type: count, tfidf, hashing"),
        train_file_path: str = typer.Argument("data/train.csv",
                                              help="Wikipedia URLs csv file path"),
        output_model_path: str = typer.Argument("vectorizer_model.pkl",
                                                help="Path to save the model")
):
    # todo add vectorizer params
    train_and_save_vectorizer(output_model_path, train_file_path, vectorizer_type)
    typer.echo(f"Model {vectorizer_type} saved as {output_model_path}.")


@cli_app.command()
def pick_best(
        query_url: str = typer.Argument(..., help="Query URL to find the best match for"),
        test_file_path: str = typer.Argument("data/test.csv",
                                             help="Path to the CSV file with test document URLs"),
        vectorizer_model_file_path: str = typer.Argument("vectorizer_model.pkl",
                                                         help="Path to the saved vectorizer model"),
        distance_metric: str = typer.Argument("cosine",
                                              help="Distance metric: cosine, euclidean, manhattan")
):
    # todo: the program additionally supports providing the entire set of queries as input to a file.
    # todo validate if test_file_path is not empty
    # todo validate if vectorizer_model_file_path is not empty
    best_match = load_vectorizer_and_pick_best(distance_metric, query_url, test_file_path, vectorizer_model_file_path)
    # todo: A list of links if the entire set of queries is provided as input.
    typer.echo(f"Best match: {best_match}")


if __name__ == "__main__":
    cli_app()
