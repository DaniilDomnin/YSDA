from copy import deepcopy

from . import Graph, operations


def word_count_graph(input_stream_name: str,
                     text_column: str = 'text',
                     count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str,
                         doc_column: str = 'doc_id',
                         text_column: str = 'text',
                         result_column: str = 'tf_idf') -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    g1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    g2 = Graph.graph_from_iter(input_stream_name).reduce(
        operations.CountRows('doc_count'), [doc_column]).map(
        operations.Project(["doc_count"]))

    g3 = deepcopy(g1).sort([doc_column, text_column]).reduce(
        operations.FirstReducer(), [doc_column, text_column]).sort(
        [text_column]).reduce(operations.Count('count'), [text_column]).join(
        operations.InnerJoiner(), g2, []).map(
        operations.Idf('doc_count', 'count'))

    g4 = deepcopy(g1).sort([doc_column]).reduce(
        operations.TermFrequency(text_column, 'tf'), [doc_column]).sort(
        [text_column]).join(
        operations.InnerJoiner(), g3, [text_column]).map(
        operations.Product(['tf', 'idf'], result_column)).sort(
        [text_column]).reduce(
        operations.TopN(result_column, 3), [text_column]).map(
        operations.Project(
            [doc_column, text_column, result_column])).sort([doc_column])

    return g4


def pmi_graph(input_stream_name: str,
              doc_column: str = 'doc_id',
              text_column: str = 'text',
              result_column: str = 'pmi') -> Graph:
    g1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)).sort(
        [doc_column, text_column]).reduce(
        operations.Count("count"), [doc_column, text_column]).map(
        operations.Filter(
            lambda x: x['count'] >= 2 and len(x[text_column]) > 4))

    g2 = deepcopy(g1).reduce(operations.SumOfAllTable
                             ('count', 'f_table'), [doc_column]).map(
        operations.Project(["f_table"]))

    g_t = deepcopy(g1).reduce(operations.Sum('count'), [doc_column])
    g3 = deepcopy(g1).join(operations.InnerJoiner(), g_t, [doc_column]).map(
        operations.Divide('count_1', 'count_2', "freq")).join(
        operations.InnerJoiner(), g2, []).join(
        operations.InnerJoiner(), g1, [doc_column, text_column]).sort(
        [text_column])  # (freq, f_tabel, count, text_column, doc_column)
    #
    g4 = deepcopy(g1).sort([text_column]).reduce(
        operations.Sum('count'), [text_column]).join(
        operations.InnerJoiner(), g3, [text_column]).map(
        operations.Divide('count_1', 'f_table', "freq_in_all")).map(
        operations.Pmi('freq', 'freq_in_all')).sort(
        [doc_column]).reduce(
        operations.TopN(result_column, 10), [doc_column]).map(
        operations.Project([doc_column, text_column, result_column]))

    return g4


def yandex_maps_graph(input_stream_name_time: str,
                      input_stream_name_length: str,
                      enter_time_column: str = 'enter_time',
                      leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id',
                      start_coord_column: str = 'start',
                      end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday',
                      hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures
    average speed in km/h depending on the weekday and hour"""

    g_length = Graph.graph_from_iter(input_stream_name_length).map(
        operations.Haversine(start_coord_column, end_coord_column, "dis")).map(
        operations.Project([edge_id_column, "dis"])).sort([edge_id_column])

    g_times = ((Graph.graph_from_iter(input_stream_name_time).map(
        operations.Time(
            leave_time_column, "%Y%m%dT%H%M%S.%f", "end_time")).map(
        operations.Time(
            enter_time_column, "%Y%m%dT%H%M%S.%f", "start_time")).map(
        operations.WeekAndHour("end_time")).map(
        operations.Project(["start_time",
                            "end_time",
                            edge_id_column,
                            hour_result_column,
                            weekday_result_column]))).map(
        operations.Minus("end_time",
                         "start_time",
                         "delta"))
               .sort([edge_id_column]).join(operations.InnerJoiner(),
                                            g_length,
                                            [edge_id_column]).
               reduce(operations.MulSum(['dis', "delta"]),
                      [weekday_result_column,
                       hour_result_column]).map(
        operations.Speed("sum_0", "sum_1", speed_result_column)).map(
        operations.Project(
            [speed_result_column, hour_result_column,
             weekday_result_column])).reduce(
        operations.FirstReducer(),
        [weekday_result_column, hour_result_column]).sort(
        [weekday_result_column, hour_result_column]).
               reduce(operations.FirstReducer(),
                      [weekday_result_column, hour_result_column]))

    return g_times
