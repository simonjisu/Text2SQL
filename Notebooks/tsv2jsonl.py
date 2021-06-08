import jsonlines
from tqdm import tqdm
from pathlib import Path
from dbengine import DBEngine


def main(sql_file_path):
    db_path = Path("./private")
    sql_gen = DBEngine(db_path=db_path / "samsung_new.db")

    # TSV file
    tsv_file = Path(sql_file_path)

    with tsv_file.open("r", encoding="utf-8") as file:
        data = [line.strip() for line in file.readlines()]
    data = [line.split("\t") for line in data]

    # SQL & NL
    file_name = tsv_file.name.split(".tsv")[0]
    sql_file = Path(file_name + ".jsonl")

    with jsonlines.Writer(sql_file.open("w")) as file:
        for line in tqdm(data, desc=sql_file.name, total=len(data)):
            question, sql, answer = line
            file.write(sql_gen.to_jsonl_sql(sql, question))

    # Table
    table_file_name = "table_" + file_name.split("_")[1] + ".jsonl"
    table_file = Path(table_file_name)
    table = sql_gen.to_jsonl_table("receipts")

    with jsonlines.Writer(table_file.open("w")) as file:
        file.write(table)

if __name__ == "__main__":
    main("./NLSQL_train.tsv")
    main("./NLSQL_test.tsv")