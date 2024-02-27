import heapq
import math
import string
from abc import abstractmethod, ABC
import typing as tp
from copy import copy
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any,
                 **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str,
                 parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any,
                 **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                line = line.strip()
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any,
                 **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable,
                 *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            for row_mapper in self.mapper(row):
                yield row_mapper


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable,
                 *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in self.reducer(tuple(self.keys), rows):
            yield row


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str],
                 rows_a: TRowsIterable,
                 rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable,
                 *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        rows_a = rows
        rows_b = args[0]
        if len(self.keys) > 0:
            key_item_a = groupby(rows_a,
                                 key=lambda x: itemgetter(*self.keys)(x))
            key_items_b = (iter
                           (groupby
                            (rows_b,
                             key=lambda x: itemgetter(*self.keys)(x))))
        else:
            joiner = CrossJoin()
            for el in joiner(self.keys, rows_a, rows_b):
                yield el
            return
        row_b: tuple[tp.Any, tp.Iterator[tp.Any]] | None = next(key_items_b)
        for key, group_items in key_item_a:
            while row_b is not None and row_b[0] < key:
                for el in self.joiner(self.keys, [], row_b[1]):
                    yield el
                try:
                    row_b = next(key_items_b)
                except StopIteration:
                    row_b = None

            if row_b is not None and row_b[0] == key:
                for el in self.joiner(self.keys, group_items, row_b[1]):
                    yield el
                try:
                    row_b = next(key_items_b)
                except StopIteration:
                    row_b = None
                continue

            for el in self.joiner(self.keys, group_items, []):
                yield el

        while row_b is not None:
            for el in self.joiner(self.keys, [], row_b[1]):
                yield el
            try:
                row_b = next(key_items_b)
            except StopIteration:
                row_b = None


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        for key, group_items in (
                groupby(rows, key=lambda x: itemgetter(*group_key)(x))):
            for el in group_items:
                yield el
                break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = (row[self.column].translate
                            (str.maketrans('', '', string.punctuation)))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = LowerCase._lower_case(row[self.column])
        yield row


class Divide(Mapper):
    """Divide columns"""

    def __init__(self, dividend_column: str,
                 divisor_column: str, res_column: str):
        """
        :param column: name of column to process
        """
        self.dividend_column = dividend_column
        self.divisor_column = divisor_column
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.res_column] = (row[self.dividend_column]
                                / row[self.divisor_column])
        yield row


class Idf(Mapper):
    """Idf count"""

    def __init__(self, column_count_documents: str, column_other: str):
        """
        :param column: name of column to process
        """
        self.column_count_documents = column_count_documents
        self.column_other = column_other

    def __call__(self, row: TRow) -> TRowsGenerator:
        row['idf'] = math.log(row[self.column_count_documents]
                              / row[self.column_other])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = [separator]
        if separator is None:
            self.separator = [' ', '\t', '\n', '\xa0']

    def __call__(self, row: TRow) -> TRowsGenerator:
        text = ""
        for sym in row[self.column]:
            if sym not in self.separator:
                text += sym
            else:
                if text != "":
                    res_row = copy(row)
                    res_row[self.column] = text
                    text = ""
                    yield res_row
                else:
                    continue

        if text != "":
            res_row = copy(row)
            res_row[self.column] = text
            yield res_row
            return

        yield None  # type: ignore


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str],
                 result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        res: tp.Any = None
        for column in self.columns:
            if res is None:
                res = row[column]
                continue
            res *= row[column]

        row[self.result_column] = res
        yield row


class Haversine(Mapper):
    """Calculate Haversine distance"""

    def __init__(self, start_point: str,
                 end_point: str, res_column: str) -> None:
        """
        :param start_point: start point(lon, lat)
        :param end_point: end point(lon, lat)
        :param res_column: result column
        """
        self.start_point = start_point
        self.end_point = end_point
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        lon1, lat1, lon2, lat2 = map(math.radians,
                                     [row[self.start_point][0],
                                      row[self.start_point][1],
                                      row[self.end_point][0],
                                      row[self.end_point][1]])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (math.sin(dlat / 2) ** 2 + math.cos(lat1)
             * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        r = 6373
        row[self.res_column] = c * r
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row
        else:
            return


class WeekAndHour(Mapper):
    """Add weekday and hour from datetime column"""

    def __init__(self, column: str) -> None:
        """
        :param column: datetime column
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        dt: dict[int, str] = dict()
        dt[0] = 'Mon'
        dt[1] = "Tue"
        dt[2] = "Wed"
        dt[3] = "Thu"
        dt[4] = "Fri"
        dt[5] = "Sat"
        dt[6] = "Sun"

        row["weekday"] = dt[row[self.column].weekday()]
        row["hour"] = row[self.column].hour
        yield row


class Speed(Mapper):
    """Cal speed in km/h"""

    def __init__(self, kil: str, time: str, res_column: str) -> None:
        """
        :param kil: distance column
        :param time: time column
        :param res_column: result column
        """
        self.kil = kil
        self.time = time
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        full_time = 0.0
        obj: timedelta = row[self.time]
        full_time += (obj.days * 24 + obj.seconds * 0.000277778 +
                      obj.microseconds * 2.7777777777778e-10)
        row[self.res_column] = row[self.kil] / full_time
        yield row


class Time(Mapper):
    """Convert str to time"""

    def __init__(self, column: str, fmt: str, res_column: str) -> None:
        """
        :param column: time colum
        :param fmt: format column
        :param res_column: result column
        """
        self.column = column
        self.fmt = fmt
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.res_column] = datetime.strptime(row[self.column], self.fmt)
        yield row


class Minus(Mapper):
    """column a minus column b"""

    def __init__(self, a: str, b: str, res_column: str) -> None:
        """
        :param a: first number
        :param b: second number
        :param res_column: result column
        """
        self.a = a
        self.b = b
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.res_column] = row[self.a] - row[self.b]
        yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = copy(row)
        for key in row.keys():
            if key not in self.columns:
                res.pop(key, None)

        yield res


class Pmi(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, column_freq_in_docj: str,
                 column_freq_ind_all: str) -> None:
        """
        :param columns: names of columns
        """
        self.column_freq_in_docj = column_freq_in_docj
        self.column_freq_ind_all = column_freq_ind_all

    def __call__(self, row: TRow) -> TRowsGenerator:
        row['pmi'] = math.log(row[self.column_freq_in_docj]
                              / row[self.column_freq_ind_all])
        yield row


# Reducers


class TopN(Reducer):
    class ComparableDict:
        def __init__(self, column: str, dt: dict[str, tp.Any]):
            self.column = column
            self.dt = dt

        def __lt__(self, other: tp.Any) -> tp.Any:
            return self.dt[self.column] < other.dt[other.column]

        def __gt__(self, other: tp.Any) -> tp.Any:
            return self.dt[self.column] > other.dt[other.column]

        def get_dict(self) -> dict[str, tp.Any]:
            return self.dt

    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        for key, group_items in (
                groupby(rows, key=lambda x: itemgetter(*group_key)(x))):
            arr: list[TopN.ComparableDict] = []
            for el in group_items:
                heapq.heappush(arr, TopN.ComparableDict(self.column_max, el))
                if len(arr) > self.n:
                    heapq.heappop(arr)

            for el in reversed(arr):  # type: ignore
                yield el.get_dict()  # type: ignore


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        for key, group_items in (
                groupby(rows, key=lambda x: itemgetter(*group_key)(x))):
            rows_dict: dict[str, TRow] = {}
            length = 0
            for el in group_items:
                length += 1
                if el[self.words_column] in rows_dict:
                    rows_dict[el[self.words_column]][self.result_column] += 1
                else:
                    new_el = copy(el)
                    for column in el.keys():
                        if (column not in group_key and
                                column != self.words_column):
                            new_el.pop(column, None)
                    new_el[self.result_column] = 1
                    rows_dict[el[self.words_column]] = new_el

            for row in rows_dict.values():
                row[self.result_column] /= length
                yield row


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        for key, group_items in (
                groupby(rows, key=lambda x: itemgetter(*group_key)(x))):
            first_el = next(group_items)
            new_el = copy(first_el)

            for column in first_el.keys():
                if column not in group_key:
                    new_el.pop(column, None)

            length = 1
            for _ in group_items:
                length += 1

            new_el[self.column] = length
            yield new_el


class CountRows(Reducer):
    """
    Count rows in a table
    """

    def __init__(self, result_column: str) -> None:
        """
        :param column: name for result column
        """
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        length = 0
        new_row = None
        for el in rows:
            if length == 0:
                new_row = copy(el)
            length += 1
        if new_row is not None:
            new_row[self.result_column] = length
            yield new_row
        else:
            return


class SumOfAllTable(Reducer):
    """
    Sums the value of a column across the entire table
    """

    def __init__(self, colum: str, res_colum: str = "sum") -> None:
        """
        :param column: name for agg column
        :param res_colum: name for res column
        """
        self.colum = colum
        self.res_column = res_colum

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        new_row = None
        for el in rows:
            if new_row is None:
                new_row = copy(el)
                new_row[self.res_column] = el[self.colum]
                continue
            new_row[self.res_column] += el[self.colum]  # type:ignore

        if new_row is not None:
            yield new_row
        else:
            return


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        for key, group_items in (
                groupby(rows, key=lambda x: itemgetter(*group_key)(x))):
            first_el = next(group_items)
            new_el = copy(first_el)

            for column in first_el.keys():
                if column not in group_key:
                    new_el.pop(column, None)

            sum: tp.Any = first_el[self.column]
            for el in group_items:
                sum += el[self.column]

            new_el[self.column] = sum
            yield new_el


class MulSum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and columns=('b', 'c')
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5, 'c': 9}
    """

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names for sum columns
        """
        self.columns = columns

    def __call__(self, group_key: tuple[str, ...],
                 rows: TRowsIterable) -> TRowsGenerator:
        for key, group_items in (
                groupby(rows, key=lambda x: itemgetter(*group_key)(x))):
            first_el = next(group_items)
            new_el = copy(first_el)

            for column in first_el.keys():
                if column not in group_key:
                    new_el.pop(column, None)

            sums: list[tp.Any] = []

            for column in self.columns:
                sums.append(first_el[column])

            for el in group_items:
                for ind, column in enumerate(self.columns):
                    sums[ind] += el[column]

            for ind, obj in enumerate(sums):
                new_el[f'sum_{ind}'] = obj

            yield new_el


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str],
                 rows_a: TRowsIterable,
                 rows_b: TRowsIterable) -> TRowsGenerator:
        list_b: list[TRow] = []
        for el in rows_b:
            list_b.append(el)
        for row_a in rows_a:
            for row_b in list_b:
                new_row = copy(row_a)
                for key, value in row_b.items():
                    if key in keys:
                        continue

                    if key in new_row:
                        new_row.pop(key, None)
                        new_row[key + self._a_suffix] = row_a[key]
                        new_row[key + self._b_suffix] = row_b[key]
                        continue
                    new_row[key] = value
                yield new_row


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str],
                 rows_a: TRowsIterable,
                 rows_b: TRowsIterable) -> TRowsGenerator:
        list_b: list[TRow] = []
        for el in rows_b:
            list_b.append(el)

        if len(list_b) == 0:
            yield from rows_a
            return

        empty = True
        for row_a in rows_a:
            empty = False
            for row_b in list_b:
                new_row = copy(row_a)
                for key, value in row_b.items():
                    if key in keys:
                        continue

                    if key in new_row:
                        new_row.pop(key, None)
                        new_row[key + self._a_suffix] = row_a[key]
                        new_row[key + self._b_suffix] = row_b[key]
                        continue
                    new_row[key] = value
                yield new_row

        if empty:
            yield from list_b


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str],
                 rows_a: TRowsIterable,
                 rows_b: TRowsIterable) -> TRowsGenerator:
        list_b: list[TRow] = []
        for el in rows_b:
            list_b.append(el)

        if len(list_b) == 0:
            yield from rows_a
            return

        for row_a in rows_a:
            for row_b in list_b:
                new_row = copy(row_a)
                for key, value in row_b.items():
                    if key in keys:
                        continue

                    if key in new_row:
                        new_row.pop(key, None)
                        new_row[key + self._a_suffix] = row_a[key]
                        new_row[key + self._b_suffix] = row_b[key]
                        continue
                    new_row[key] = value
                yield new_row


class CrossJoin(Joiner):
    def __call__(self, keys: tp.Sequence[str],
                 rows_a: TRowsIterable,
                 rows_b: TRowsIterable) -> TRowsGenerator:
        list_b: list[TRow] = []
        for el in rows_b:
            list_b.append(el)
        for row_a in rows_a:
            for row_b in list_b:
                new_row = copy(row_a)
                for key, value in row_b.items():
                    if key in new_row:
                        new_row.pop(key, None)
                        new_row[key + self._a_suffix] = row_a[key]
                        new_row[key + self._b_suffix] = row_b[key]
                        continue
                    new_row[key] = value
                yield new_row


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str],
                 rows_a: TRowsIterable,
                 rows_b: TRowsIterable) -> TRowsGenerator:
        list_a: list[TRow] = []
        for el in rows_a:
            list_a.append(el)

        if len(list_a) == 0:
            yield from rows_b
            return

        for row_b in rows_b:
            for row_a in list_a:
                new_row = copy(row_b)
                for key, value in row_a.items():
                    if key in keys:
                        continue

                    if key in new_row:
                        new_row.pop(key, None)
                        new_row[key + self._a_suffix] = row_a[key]
                        new_row[key + self._b_suffix] = row_b[key]
                        continue
                    new_row[key] = value
                yield new_row
