import typing as tp
from compgraph import graph, operations


def test_read_from_file(tmp_path: tp.Any) -> None:
    def compare(expect: list[dict[str, tp.Any]],
                real: list[dict[str, tp.Any]]) -> None:
        assert expect == real

    def parser_one(line: str) -> dict[str, tp.Any]:
        res: dict[str, tp.Any] = dict()
        for token in line.split('$'):
            key, value = token.split(':')
            if str.isnumeric(value):
                new_val = int(value)
                res[key] = new_val
                continue
            res[key] = value
        return res

    def parser_two(line: str) -> dict[str, tp.Any]:
        res: dict[str, tp.Any] = dict()
        for token in line.split('$'):
            key, value = token.split(':')
            if value[0] == '[':
                value = value.translate(str.maketrans('', '', '[],'))
                values = [float(x) for x in value.split(' ')]
                res[key] = values
                continue
            if str.isnumeric(value):
                new_val = int(value)
                res[key] = new_val
                continue
            res[key] = value
        return res

    d = tmp_path / "inputs_test"
    d.mkdir()
    p_1 = d / "test_read_from_file.txt"
    p_2 = d / "test_read_from_file_2.txt"
    text_1 = ("doc_id:1$text:hello, little world\ndoc_id:2$text:little\ndoc_id:3$text:little little "  # noqa: E501
              "little\ndoc_id:4$text:little? hello little world\ndoc_id:5$text:HELLO HELLO! "  # noqa: E501
              "WORLD...\ndoc_id:6$text:world? world... world!!! WORLD!!! HELLO!!!")  # noqa: E501
    text_2 = ("start:[37.84870228730142, 55.73853974696249]$end:[37.8490418381989, "  # noqa: E501
              "55.73832445777953]$edge_id:8414926848168493057\nstart:[37.524768467992544, 55.88785375468433]$end:["  # noqa: E501
              "37.52415172755718, 55.88807155843824]$edge_id:5342768494149337085\nstart:[37.56963176652789, "  # noqa: E501
              "55.846845586784184]$end:[37.57018438540399, 55.8469259692356]$edge_id:5123042926973124604")  # noqa: E501
    p_1.write_text(text_1, encoding="utf-8")
    p_2.write_text(text_2, encoding="utf-8")
    cases: list[list[tp.Any]] = [
        [
            f"{tmp_path}/inputs_test/test_read_from_file.txt",
            parser_one,
            [{'doc_id': 1, 'text': 'hello, little world'},
             {'doc_id': 2, 'text': 'little'},
             {'doc_id': 3, 'text': 'little little little'},
             {'doc_id': 4, 'text': 'little? hello little world'},
             {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
             {'doc_id': 6,
              'text': 'world? world... world!!! WORLD!!! HELLO!!!'}]
        ],
        [
            f"{tmp_path}/inputs_test/test_read_from_file_2.txt",
            parser_two,
            [
                {'start': [37.84870228730142, 55.73853974696249],
                 'end': [37.8490418381989, 55.73832445777953],
                 'edge_id': 8414926848168493057},
                {'start': [37.524768467992544, 55.88785375468433],
                 'end': [37.52415172755718, 55.88807155843824],
                 'edge_id': 5342768494149337085},
                {'start': [37.56963176652789, 55.846845586784184],
                 'end': [37.57018438540399, 55.8469259692356],
                 'edge_id': 5123042926973124604}
            ]
        ]
    ]

    for case in cases:
        g = graph.Graph.graph_from_file(case[0], case[1])
        res: list[dict[str, tp.Any]] = []
        for row in g.run():
            res.append(row)
        compare(case[2], res)


def test_consecutive() -> None:
    def compare(expect: list[dict[str, tp.Any]],
                real: list[dict[str, tp.Any]]) -> None:
        assert expect == real

    rows_a_1 = [{'count': 14, 'mass': 7},
                {'count': 3, 'mass': 18},
                {'count': 10, 'mass': 3},
                ]

    rows_a_2 = [{'count': 1, 'mass': 2},
                {'count': 3, 'mass': 4},
                ]

    rows_b_1_a = [{'id': 1, 'count': 12, 'mass': 15, },
                  {'id': 2, 'count': 6, 'mass': 7},
                  ]

    rows_b_2_a = [{'id': 1, 'height': 8},
                  {'id': 2, 'height': 15},
                  ]

    rows_b_1_b = [{'id': 3, 'count': 15, 'mass': 4, },
                  {'id': 4, 'count': 18, 'mass': 3},
                  ]

    rows_b_2_b = [{'id': 3, 'height': 9},
                  {'id': 4, 'height': 13},
                  ]

    expected_1 = [{'count': 14, 'full': 98, 'mass': 7, 'sum': 182}]
    expected_2 = [{'count': 1, 'full': 2, 'mass': 2, 'sum': 14}]
    expected_3 = [{'id': 1, 'count': 12, 'mass': 15, 'height': 8},
                  {'id': 2, 'count': 6, 'mass': 7, 'height': 15},
                  ]
    expected_4 = [{'id': 3, 'count': 15, 'mass': 4, 'height': 9},
                  {'id': 4, 'count': 18, 'mass': 3, 'height': 13},
                  ]

    # basic test
    g1 = (graph.Graph.graph_from_iter("texts").map(
        operations.Product(['count', 'mass'], 'full')).reduce(
        operations.SumOfAllTable('full', 'sum'), []))

    res: list[dict[str, tp.Any]] = []
    for row in g1.run(texts=lambda: iter(rows_a_1)):
        res.append(row)

    compare(expected_1, res)
    res.clear()
    for row in g1.run(texts=lambda: iter(rows_a_2)):
        res.append(row)
    compare(expected_2, res)
    res.clear()

    # join test
    g1 = graph.Graph.graph_from_iter("height")
    g2 = (graph.Graph.graph_from_iter("count_mass").
          join(operations.InnerJoiner(), g1, ['id']))
    for row in g2.run(height=lambda: iter(rows_b_2_a),
                      count_mass=lambda: iter(rows_b_1_a)):
        res.append(row)
    compare(expected_3, res)

    res.clear()
    for row in g2.run(height=lambda: iter(rows_b_2_b),
                      count_mass=lambda: iter(rows_b_1_b)):
        res.append(row)
    compare(expected_4, res)
