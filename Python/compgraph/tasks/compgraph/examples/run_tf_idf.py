import json

import click
from compgraph.algorithms import inverted_index_graph


# TODO: cli
# You can use anything you want. We suggest you to use `click`
@click.command()
@click.argument('input_filepath')
@click.argument("output_filepath")
def main(input_filepath: str, output_filepath: str) -> None:
    graph = inverted_index_graph('texts', doc_column='doc_id', text_column='text', result_column='tf_idf')
    rows = []
    with open(input_filepath, 'r') as inp:
        for line in inp:
            line = line.strip()
            rows.append(json.loads(line))

    result = graph.run(texts=lambda: iter(rows))
    with open(output_filepath, "w") as out:
        for row in result:
            json.dump(row, out)
            out.write('\n')


if __name__ == "__main__":
    main()
