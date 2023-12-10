import redis
import hashlib
import pyarrow as pa

from datetime import timedelta


class RedisCache:
    """We are wrapping the redis client to provide a stable interface
    in case we want to switch the caching solution."""

    def __init__(self):
        # TODO: Remove this hardcoded server value, add to config
        self.connection = redis.Redis(host="redis-stack-server", port=6379)

    def set_json(self, key, value, ttl=timedelta(days=7)):
        # We are using query strings as keys, better to hash them for perf
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()

        self.connection.json().set(key, "$", value)
        self.connection.expire(key, ttl)

    def get_json(self, key):
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.connection.json().get(key)

    def set_dataframe(self, key, dataframe, ttl=timedelta(days=7)):
        if dataframe is not None:
            self.connection.set(
                key, self.serialize_pandas(dataframe, preserve_index=True).to_pybytes()
            )
            self.connection.expire(key, ttl)

    def get_dataframe(self, key):
        data = self.connection.get(key)
        return self.deserialize_pandas(data) if data is not None else None

    def is_available(self):
        try:
            self.connection.ping()
        except (redis.exceptions.ConnectionError, ConnectionRefusedError):
            return False
        return True

    # Copy-pasted from pyarrow source code and modified to work around
    # https://github.com/apache/arrow/issues/38489
    # by manually modifying the schema
    def serialize_pandas(self, df, *, nthreads=None, preserve_index=None):
        schema = pa.Schema.from_pandas(df, preserve_index=preserve_index)

        # FIXME: Probably not good idea to hardcode this
        if "ranks" in df.columns:
            schema = schema.set(
                df.columns.get_loc("ranks"),
                pa.field("ranks", pa.map_(pa.string(), pa.float64(), keys_sorted=True)),
            )

        batch = pa.RecordBatch.from_pandas(
            df, schema, nthreads=nthreads, preserve_index=preserve_index
        )
        sink = pa.BufferOutputStream()
        with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
            writer.write_batch(batch)
        return sink.getvalue()

    # Copy-pasted from pyarrow source code and modified to work around
    # https://github.com/apache/arrow/issues/38489
    # maps_as_pydicts="srict" is the relevant change
    def deserialize_pandas(self, buf, *, use_threads=True):
        buffer_reader = pa.BufferReader(buf)
        with pa.RecordBatchStreamReader(buffer_reader) as reader:
            table = reader.read_all()
        return table.to_pandas(use_threads=use_threads, maps_as_pydicts="strict")
