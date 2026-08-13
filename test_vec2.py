import sqlite3
import sqlite_vec
import struct

db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vec.load(db)

db.execute("CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])")
db.execute("CREATE TABLE meta (rowid INTEGER PRIMARY KEY, category TEXT)")

db.execute("INSERT INTO vec_test(rowid, embedding) VALUES (1, ?)", [struct.pack('%sf' % 3, *[1.0, 0.0, 0.0])])
db.execute("INSERT INTO meta(rowid, category) VALUES (1, 'A')")

db.execute("INSERT INTO vec_test(rowid, embedding) VALUES (2, ?)", [struct.pack('%sf' % 3, *[0.0, 1.0, 0.0])])
db.execute("INSERT INTO meta(rowid, category) VALUES (2, 'B')")

db.execute("INSERT INTO vec_test(rowid, embedding) VALUES (3, ?)", [struct.pack('%sf' % 3, *[1.1, 0.0, 0.0])])
db.execute("INSERT INTO meta(rowid, category) VALUES (3, 'B')")

query = struct.pack('%sf' % 3, *[1.0, 0.0, 0.0])

# Explicit distance with join
sql = '''
SELECT v.rowid, vec_distance_L2(v.embedding, ?) as distance
FROM vec_test v
INNER JOIN meta m ON v.rowid = m.rowid
WHERE m.category = 'B'
ORDER BY distance ASC
LIMIT 2
'''
rows = db.execute(sql, [query]).fetchall()
print("Explicit distance results:", rows)
