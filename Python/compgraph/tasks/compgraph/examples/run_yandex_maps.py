import json
from itertools import islice, cycle

import click
from compgraph.algorithms import yandex_maps_graph


# TODO: cli
# You can use anything you want. We suggest you to use `click`
@click.command()
@click.argument('input_length')
@click.argument('input_times')
@click.argument("output_filepath")
def main(input_length: str, input_times: str, output_filepath: str) -> None:
    graph = yandex_maps_graph(
        'travel_time', 'edge_length',
        enter_time_column='enter_time', leave_time_column='leave_time', edge_id_column='edge_id',
        start_coord_column='start', end_coord_column='end',
        weekday_result_column='weekday', hour_result_column='hour', speed_result_column='speed'
    )
    times = []
    lengths = []
    with open(input_length, 'r') as inp:
        for line in inp:
            line = line.strip()
            lengths.append(json.loads(line))

    with open(input_times, 'r') as inp:
        for line in inp:
            line = line.strip()
            times.append(json.loads(line))

    result = graph.run(travel_time=lambda: islice(cycle(iter(times)), len(times)), edge_length=lambda: iter(lengths))
    with open(output_filepath, "w") as out:
        for row in result:
            json.dump(row, out)
            out.write('\n')


if __name__ == "__main__":
    main()
