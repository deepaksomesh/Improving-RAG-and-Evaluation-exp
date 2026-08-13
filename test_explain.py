import sqlite3
import sqlite_vec
import struct

db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vec.load(db)

db.execute("CREATE VIRTUAL TABLE chunk_vec USING vec0(embedding float[3])")
db.execute("CREATE TABLE chunk_metadata(rowid INTEGER PRIMARY KEY, document_id TEXT)")
db.execute("CREATE INDEX idx_document_id ON chunk_metadata(document_id)")

query = struct.pack('%sf' % 3, *[1.0, 0.0, 0.0])

print("Explain with 3 items:")
for row in db.execute("EXPLAIN QUERY PLAN SELECT v.rowid, vec_distance_L2(v.embedding, ?) as distance FROM chunk_metadata m INNER JOIN chunk_vec v ON m.rowid = v.rowid WHERE m.document_id IN (?, ?, ?) ORDER BY distance ASC LIMIT 5", [query, "doc1", "doc2", "doc3"]).fetchall():
    print(row)
