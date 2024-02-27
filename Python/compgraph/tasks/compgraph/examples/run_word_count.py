import json

import click
from compgraph.algorithms import word_count_graph


# TODO: cli
# You can use anything you want. We suggest you to use `click`
@click.command()
@click.argument('input_filepath')
@click.argument("output_filepath")
def main(input_filepath: str, output_filepath: str) -> None:
    graph = word_count_graph(input_stream_name="input", text_column='text', count_column='count')
    rows = []
    with open(input_filepath, 'r') as inp:
        for line in inp:
            line = line.strip()
            rows.append(json.loads(line))

    result = graph.run(input=lambda: iter(rows))
    with open(output_filepath, "w") as out:
        for row in result:
            json.dump(row, out)
            out.write('\n')


if __name__ == "__main__":
    main()
