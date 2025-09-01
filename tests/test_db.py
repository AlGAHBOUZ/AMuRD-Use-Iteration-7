import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.modules.db import TeradataDatabase


if __name__ == "__main__":
    db = TeradataDatabase()
    db.connect()
    print(20*"-", "Execute Query", 20*"-")
    query = """SELECT * FROM demo_user.products SAMPLE 10;"""
    tdf = db.execute_query(query)
    print(tdf)