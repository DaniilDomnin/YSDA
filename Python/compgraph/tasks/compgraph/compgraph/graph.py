import typing as tp
from . import operations as ops
from . import external_sort


class Graph:
    """Computational graph implementation """

    VAR = 0

    def __init__(self) -> None:
        self.rows = None
        self.Operations_sequence: list[tp.Any] = []
        self.joiners: list['Graph'] = []
        self.i = 0

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data
         from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        g = Graph()
        g.Operations_sequence.append(ops.ReadIterFactory(name))
        return g

    @staticmethod
    def graph_from_file(filename: str,
                        parser:
                        tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation
         for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        g = Graph()
        g.Operations_sequence.append(ops.Read(filename, parser))
        return g

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map
         operation with particular mapper
        :param mapper: mapper to use
        """
        self.Operations_sequence.append(ops.Map(mapper))
        return self

    def reduce(self, reducer: ops.Reducer,
               keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce
         operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        self.Operations_sequence.append(ops.Reduce(reducer, keys))
        return self

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        sort = external_sort.ExternalSort(keys)
        self.Operations_sequence.append(sort)
        return self

    def join(self, joiner: ops.Joiner,
             join_graph: 'Graph',
             keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        self.Operations_sequence.append(ops.Join(joiner, keys))
        self.joiners.append(join_graph)
        return self

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        self.i = 0
        self.rows = None
        """Single method to start execution; data sources passed as kwargs"""
        for ind, operation in enumerate(self.Operations_sequence):
            if isinstance(operation, ops.ReadIterFactory):
                self.rows = operation(**kwargs)  # type:ignore
                continue
            elif isinstance(operation, ops.Read):
                self.rows = operation()  # type:ignore
                continue
            elif isinstance(operation, ops.Join):
                cur_joiner = self.joiners[self.i]
                self.i += 1
                if self.rows is not None:
                    self.rows = operation(  # type:ignore
                        self.rows, cur_joiner.run(**kwargs))
                continue

            self.rows = operation(self.rows)

        if self.rows is not None:
            return iter(self.rows)  # type:ignore
        else:
            assert False
