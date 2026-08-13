import sqlite3
import sqlite_vec
import struct

db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vec.load(db)

db.execute("CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[3])")
db.execute("INSERT INTO vec_test(rowid, embedding) VALUES (1, ?)", [struct.pack('%sf' % 3, *[1.0, 0.0, 0.0])])
db.execute("INSERT INTO vec_test(rowid, embedding) VALUES (2, ?)", [struct.pack('%sf' % 3, *[0.0, 1.0, 0.0])])

query = struct.pack('%sf' % 3, *[1.0, 0.0, 0.0])

# Using MATCH
rows = db.execute("SELECT rowid, distance FROM vec_test WHERE embedding MATCH ? LIMIT 2", [query]).fetchall()
print("MATCH results:", rows)

# Using explicit distance
rows2 = db.execute("SELECT rowid, vec_distance_L2(embedding, ?) as dist FROM vec_test ORDER BY dist ASC LIMIT 2", [query]).fetchall()
print("vec_distance_L2 results:", rows2)
