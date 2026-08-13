import sqlite3
import sqlite_vec
import time
import struct
import random

db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
db.execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[768])")

print("Inserting 64000 vectors...")
vectors = []
for i in range(64000):
    vec = [random.random() for _ in range(768)]
    vectors.append((i+1, struct.pack('%sf' % 768, *vec)))
db.executemany("INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)", vectors)
db.commit()

print("Querying...")
query_vec = struct.pack('%sf' % 768, *[random.random() for _ in range(768)])
start = time.time()
db.execute("SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance ASC LIMIT 5", [query_vec]).fetchall()
end = time.time()
print(f"Time taken for 1 query: {end - start:.5f}s")
