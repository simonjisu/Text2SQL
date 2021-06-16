import re
import records
import jsonlines
from pathlib import Path
from typing import Union
from moz_sql_parser import parse as sql_parser
schema_re = re.compile(r'\((.+)\)')

class DBEngine:
    
    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<", "OP"] #, ">=", "<="]
    cond_ops_dict = {"eq": "=", "lt": "<",  "lte": "<=", "gt": ">", "gte": ">=", "neq": "<>"}

    syms = ["SELECT", "WHERE", "AND", "COL", "TABLE", "CAPTION", "PAGE", "SECTION", "OP", "COND", "QUESTION", "AGG", "AGGOPS", "CONDOPS"]
    
    def __init__(self, db_path: Union[Path, str]) -> None:
        self.db = records.Database(f"sqlite:///{db_path}")
        self._reset()
        
    def _reset(self) -> None:
        self.table_id = None
        self.schema = None
        self.col2idx = None
        
    def get_schema_info(self, table_id: str) -> None:
        table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql
        schema_str = schema_re.findall(table_info.replace("\n", ""))[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c.strip('"')] = t
        col2idx = {c: i for i, c in enumerate(schema.keys())}
        
        self.table_id = table_id
        self.schema = schema
        self.col2idx = col2idx
    
    def to_jsonl_table(self, table_id: str) -> dict:
        self._reset()
        self.get_schema_info(table_id)
        table = {"id": self.table_id, "header": list(self.schema), "types": list(self.schema.values()), "rows": []}
        q = f'SELECT * FROM "{self.table_id}"'
        res = self.db.query(q)
        for row in res.all():
            table["rows"].append(list(row.values()))
        return table
    
    def to_jsonl_sql(self, sql: str, question: str) -> dict:
        r"""
        # Only 1 agg and select
        example:
        - sql: "SELECT frmtrm_amount FROM receipts WHERE account_nm = '이익잉여금' AND bsns_year = 2020",
        - question: "제 51 기에 삼성전자의 유동자산은 어떻게 돼?"
        
        return:
        {
           "phase":1,
           "question":"제 51 기에 삼성전자의 유동자산은 어떻게 돼?",
           "sql":{
              "conds":[
                 [10, 0, "이익잉여금"], [3, 0, 2020]
              ],
              "sel":16,
              "agg":0
           },
           "table_id":"receipts"
        }
        
        """
        
        parsed = sql_parser(sql)
        table_id = parsed["from"]
        jsonl = {"phase": 1, "question": question, "table_id": table_id, "sql": {}}
        
        if (self.table_id is None) or (self.table_id != table_id):
            self.get_schema_info(table_id)
#         else:
#             raise AttributeError("No schema information, please make sure to call `self.get_schema_info`")
        
        select_parsed = parsed["select"]["value"]
        if isinstance(select_parsed, dict):
            # Only 1 agg and select
            agg_name = list(select_parsed)[0]
            agg = self.agg_ops.index(agg_name.upper())
            select_name = select_parsed[agg]
        elif isinstance(select_parsed, str):
            agg = 0
            select_name = select_parsed
        else:
            raise TypeError(f"Parsed in select clause should be `str` or `dict` type, Current is {select_parsed}")
        select = self.col2idx.get(select_name)
        
        
        conds = []
        if parsed.get("where"):
            conds_parsed = parsed["where"]
            for operator, conditions in conds_parsed.items():
                # cond = {operator.upper(): []}
                # Is operator always AND?
                if operator == "and":
                    for condition in conditions:
                        key, values = tuple(condition.items())[0]

                        if self.cond_ops_dict.get(key) is None:
                            raise KeyError(f"No operator: {key}")
                        else:
                            op = self.cond_ops_dict.get(key)
                            op_idx = self.cond_ops.index(op)

                        if self.col2idx.get(values[0]) is None:
                            raise KeyError(f"No column name: {values[0]}")
                        else:
                            col_idx = self.col2idx.get(values[0])


                        if isinstance(values[1], dict):
                            # make sure all string values insert '' when parse to sql again
                            cond_value = values[1]["literal"]
                        else:
                            cond_value = values[1]

                        cond = [col_idx, op_idx, cond_value]
                        conds.append(cond)
                else:
                    # only contains 1 condition
                    key = operator
                    values = conditions
                    if self.cond_ops_dict.get(key) is None:
                        raise KeyError(f"No operator: {key}")
                    else:
                        op = self.cond_ops_dict.get(key)
                        op_idx = self.cond_ops.index(op)

                    if self.col2idx.get(values[0]) is None:
                        raise KeyError(f"No column name: {values[0]}")
                    else:
                        col_idx = self.col2idx.get(values[0])


                    if isinstance(values[1], dict):
                        # make sure all string values insert '' when parse to sql again
                        cond_value = values[1]["literal"]
                    else:
                        cond_value = values[1]

                    cond = [col_idx, op_idx, cond_value]
                    conds.append(cond)
                
        jsonl["sql"]["sel"] = select
        jsonl["sql"]["agg"] = agg
        jsonl["sql"]["conds"] = conds
        return jsonl
    
    def to_sql(self, sql_jsonl: dict) -> str:
        table_id = sql_jsonl["table_id"]
        if self.table_id != table_id:
            self._reset()
            self.get_schema_info(table_id)
        
        
        sql_dict = sql_jsonl["sql"]  # dict_keys(['sel', 'agg', 'conds'])
        sel_idx = sql_dict["sel"]
        col = list(self.schema)[sel_idx]
        agg_idx = sql_dict["agg"]
        agg = self.agg_ops[agg_idx]
        
        conds = sql_dict["conds"]
        cond_str = ""
        for col_idx, op_idx, cond_value in conds:
            cond_value = f"'{cond_value}'" if isinstance(cond_value, str) else cond_value
            cond_str += f"{list(self.schema)[col_idx]} {self.cond_ops[op_idx]} {cond_value}"
            cond_str += " AND "
        
        query = f"SELECT {agg}({col}) FROM {self.table_id} WHERE {cond_str}"
        return query