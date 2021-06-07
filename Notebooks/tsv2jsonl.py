import jsonlines
from pathlib import Path
from dbengine import DBEngine


def main():
    db_path = Path("./private")
    sql_gen = DBEngine(db_path=db_path / "samsung_new.db")

    # TSV file
    tsv_file = Path("./NLSQL.tsv")

    with tsv_file.open("r", encoding="utf-8") as file:
        data = [line.strip() for line in file.readlines()]
    data = [line.split("\t") for line in data]

    # SQL & NL
    sql_file = Path("./NLSQL.jsonl")

    with jsonlines.Writer(sql_file.open("w")) as file:
        for line in data:
            question, sql, answer = line
            file.write(sql_gen.to_jsonl_sql(sql, question))

    # Table
    table_file = Path("./table.jsonl")
    table = sql_gen.to_jsonl_table("receipts")

    with jsonlines.Writer(table_file.open("w")) as file:
        file.write(table)

if __name__ == "__main__":
    main()