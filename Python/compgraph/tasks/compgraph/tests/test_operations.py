import math
import typing as tp
from datetime import datetime, timedelta
from pytest import approx

from compgraph import operations


def compare_map(expect: dict[str, tp.Any],
                real: tp.Generator[dict[str, tp.Any], None, None]) -> None:
    elements: list[dict[str, tp.Any]] = []

    for el in real:
        elements.append(el)

    assert len(elements) == 1
    assert expect == elements[0]


def compare_reduce(expect: list[dict[str, tp.Any]],
                   real: tp.Generator[dict[str, tp.Any], None, None]) -> None:
    elements: list[dict[str, tp.Any]] = []

    for el in real:
        elements.append(el)

    assert expect == elements


def test_cross_join() -> None:
    expected = [{'id_1': 14, 'id_2': 14, 'key_a_1': 1, 'key_a_2': 1}]
    rows_a = [{'id': 14, 'key_a': 1}]
    rows_b = [{'id': 14, 'key_a': 1}]
    res = operations.CrossJoin()([], rows_a, rows_b)
    compare_reduce(expected, res)


def test_outer_join() -> None:
    expected = [{'id': 14, 'common_key_1': 2,
                 'common_key_2': 3,
                 'key_a': 16, 'key_b': 25}]
    rows_a = [{'id': 14, 'common_key': 2, 'key_a': 16}]
    rows_b = [{'id': 14, 'common_key': 3, 'key_b': 25}]

    res = operations.OuterJoiner()(['id'], rows_a, rows_b)
    compare_reduce(expected, res)


def test_divide() -> None:
    expected = [{'id': 14, 'a': 7, 'b': 4, "res": 7 / 4},
                {'id': 15, 'a': 15, "b": 3, "res": 15 / 3}]
    rows = [{'id': 14, 'a': 7, 'b': 4},
            {'id': 15, 'a': 15, "b": 3}]

    for ind, row in enumerate(rows):
        res = operations.Divide('a', 'b', 'res')(row)
        compare_map(expected[ind], res)


def test_idf() -> None:
    expected = [{'id': 14, 'all_count': 3,
                 "local_count": 2,
                 'idf': math.log(3 / 2)},
                {'id': 15,
                 'all_count': 7,
                 "local_count": 3,
                 'idf': math.log(7 / 3)}]
    rows = [{'id': 14, 'all_count': 3, "local_count": 2},
            {'id': 15, 'all_count': 7, "local_count": 3}]

    for ind, row in enumerate(rows):
        res = operations.Idf('all_count', 'local_count', )(row)
        compare_map(expected[ind], res)


def test_weekAndHour() -> None:
    expected = [{'time': datetime.strptime('20171020T112238.723000',
                                           "%Y%m%dT%H%M%S.%f"),
                 'weekday': 'Fri', 'hour': 11},
                {'time': datetime.strptime('20171011T145553.040000',
                                           "%Y%m%dT%H%M%S.%f"),
                 'weekday': 'Wed', 'hour': 14},
                {'time': datetime.strptime('20171020T090548.939000',
                                           "%Y%m%dT%H%M%S.%f"),
                 'weekday': 'Fri', 'hour': 9}]

    rows = [{'time': datetime.strptime('20171020T112238.723000',
                                       "%Y%m%dT%H%M%S.%f")},
            {'time': datetime.strptime('20171011T145553.040000',
                                       "%Y%m%dT%H%M%S.%f")},
            {'time': datetime.strptime('20171020T090548.939000',
                                       "%Y%m%dT%H%M%S.%f")}]

    for ind, row in enumerate(rows):
        res = operations.WeekAndHour('time')(row)
        compare_map(expected[ind], res)


def test_speed() -> None:
    expected = [{'time': timedelta(hours=3, seconds=124),
                 'dis': 15,
                 'speed': approx(15 / (3 + 124 / 60 / 60), 0.001)},
                {'time': timedelta(hours=1, seconds=54),
                 'dis': 7,
                 'speed': approx(7 / (1 + 54 / 60 / 60), 0.001)},
                {'time': timedelta(seconds=124),
                 'dis': 4,
                 'speed': approx(4 / (124 / 60 / 60), 0.001)}]

    rows = [{'time': timedelta(hours=3, seconds=124), 'dis': 15},
            {'time': timedelta(hours=1, seconds=54), 'dis': 7},
            {'time': timedelta(seconds=124), 'dis': 4}]

    for ind, row in enumerate(rows):
        res = operations.Speed(kil='dis', time='time', res_column='speed')(row)
        compare_map(expected[ind], res)


def test_time() -> None:
    expected = [
        {'time': '20171020T112238.723000',
         'res':
             datetime.strptime('20171020T112238.723000', "%Y%m%dT%H%M%S.%f")},
        {'time': '20171011T145553.040000',
         'res':
             datetime.strptime('20171011T145553.040000', "%Y%m%dT%H%M%S.%f")},
        {'time': '20171020T090548.939000',
         'res':
             datetime.strptime('20171020T090548.939000', "%Y%m%dT%H%M%S.%f")}]

    rows = [{'time': '20171020T112238.723000'},
            {'time': '20171011T145553.040000'},
            {'time': '20171020T090548.939000'}]

    for ind, row in enumerate(rows):
        res = operations.Time(
            column='time', fmt="%Y%m%dT%H%M%S.%f", res_column='res')(row)
        compare_map(expected[ind], res)


def test_minus() -> None:
    expected = [{'count': 15, 'loss': 3, 'res': 12},
                {'count': 4, 'loss': 2, 'res': 2},
                {'count': 1, 'loss': 5, 'res': -4}]

    rows = [{'count': 15, 'loss': 3},
            {'count': 4, 'loss': 2},
            {'count': 1, 'loss': 5}]

    for ind, row in enumerate(rows):
        res = operations.Minus('count', 'loss', 'res')(row)
        compare_map(expected[ind], res)


def test_pmi() -> None:
    expected = [{'freq_doc': 13, 'freq_all': 15,
                 'pmi': approx(-0.1431, 0.001)},
                {'freq_doc': 2, 'freq_all': 3,
                 'pmi': approx(-0.4054, 0.001)},
                {'freq_doc': 8, 'freq_all': 30,
                 'pmi': approx(-1.322, 0.001)}]

    rows = [{'freq_doc': 13, 'freq_all': 15},
            {'freq_doc': 2, 'freq_all': 3},
            {'freq_doc': 8, 'freq_all': 30}]

    for ind, row in enumerate(rows):
        res = operations.Pmi('freq_doc', 'freq_all')(row)
        compare_map(expected[ind], res)


def test_haversine() -> None:
    expected = [{'start': (48.43, 37.14),
                 'end': (47.34, 36.78),
                 'res': approx(104.8, 0.01)},
                {'start': (12.2, 123.4),
                 'end': (4.9, 100.45),
                 'res': approx(2565, 0.01)}]

    rows = [{'start': (48.43, 37.14), 'end': (47.34, 36.78)},
            {'start': (12.2, 123.4), 'end': (4.9, 100.45)}]

    for ind, row in enumerate(rows):
        res = operations.Haversine('start', 'end', 'res')(row)
        compare_map(expected[ind], res)


def test_CountRows() -> None:
    expected = [[{'a': 15, 'b': 45, 'c': "asd", 'count': 4}],
                [{'a': 'f', 'count': 1}],
                [{'a': 1, 'b': 1, 'c': "q", 'count': 2}
                 ]
                ]

    tests_data = [[
        {'a': 15, 'b': 45, 'c': "asd"},
        {'a': 45, 'b': 34, 'c': "qwer"},
        {'a': 13, 'b': 86, 'c': "eqw"},
        {'a': 64, 'b': 35, 'c': "asd"}
    ],
        [{'a': 'f'}],
        [{'a': 1, 'b': 1, 'c': "q"},
         {'a': 1, 'b': 1, 'c': "q"}
         ]
    ]

    for ind, case in enumerate(tests_data):
        res = operations.CountRows('count')(tuple('a'), case)  # type:ignore
        compare_reduce(expected[ind], res)


def test_SumOfAllTable() -> None:
    expected = [[{'a': 15, 'b': 45, 'c': "asd", "sum": 200}],
                [{'a': 'f', 'b': 3, 'sum': 3}],
                [{'a': 1, 'b': 1, 'c': "q", "sum": 2}
                 ]
                ]

    tests_data = [[
        {'a': 15, 'b': 45, 'c': "asd"},
        {'a': 45, 'b': 34, 'c': "qwer"},
        {'a': 13, 'b': 86, 'c': "eqw"},
        {'a': 64, 'b': 35, 'c': "asd"}
    ],
        [{'a': 'f', 'b': 3}],
        [{'a': 1, 'b': 1, 'c': "q"},
         {'a': 1, 'b': 1, 'c': "q"}
         ]
    ]

    for ind, case in enumerate(tests_data):
        res = operations.SumOfAllTable('b')(tuple('a'), case)
        compare_reduce(expected[ind], res)


def test_MulSum() -> None:
    expected = [[{'id': 1, 'sum_0': 200, 'sum_1': 10},
                 {'id': 2, 'sum_0': 4, 'sum_1': 3}],
                [{'id': 2, 'sum_0': 3, 'sum_1': 4}],
                [{'id': 1, 'sum_0': 1, 'sum_1': 3},
                 {'id': 2, 'sum_0': 4, 'sum_1': 3},
                 {'id': 3, 'sum_0': 4, 'sum_1': 3}
                 ]
                ]

    tests_data = [[
        {'id': 1, 'b': 45, 'c': 2},
        {'id': 1, 'b': 34, 'c': 3},
        {'id': 1, 'b': 86, 'c': 5},
        {'id': 1, 'b': 35, 'c': 0},
        {'id': 2, 'b': 4, 'c': 3}
    ],
        [{'id': 2, 'b': 3, 'c': 4}],
        [{'id': 1, 'b': 1, 'c': 3},
         {'id': 2, 'b': 4, 'c': 3},
         {'id': 3, 'b': 4, 'c': 3}
         ]
    ]

    for ind, case in enumerate(tests_data):
        res = operations.MulSum(['b', 'c'])(['id'], case)  # type:ignore
        compare_reduce(expected[ind], res)
