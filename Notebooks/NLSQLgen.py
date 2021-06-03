from pathlib import Path
import re
import records
from babel.numbers import parse_decimal, NumberFormatError
from pyjosa.josa import Josa 
import numpy as np 
schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')

db_path = Path("./private/db")
db = records.Database(f"sqlite:///{db_path /'samsung_new.db'}")

# From Variable to Natural Language 
def time_to_nl(time):   
    list = []
    list.append(time)
    list.append(f'{time}의')
    list.append(f'{time}년')
    list.append(f'{time}년의')
    list.append(f'{time}년도')
    list.append(f'{time}년도의')
    
    gap = 1968
    th = time - gap # 52
    list.append(f'{th}기')
    list.append(f'{th}기의')
    list.append(f'제{th}기')
    list.append(f'제{th}기의')
    return list
        
def entity_to_nl(entity):
    list = []
    list.append(entity)
    list.append(f'{entity}의')
    return list 
def account_to_nl(account):
    list = []
    list.append(account + Josa.get_josa(account,'은'))
    list.append(account + Josa.get_josa(account,'가'))
    return list
def question_to_nl():
    return ['어떻게 돼?', '얼마야?', '어때?', '몇이야?']

def key_to_nl(time, entity, account):
    lists = []
    for t in time_to_nl(time):
        for e in entity_to_nl(entity):
            for a in account_to_nl(account):
                for q in question_to_nl():
                    
                    lists.append(f'{t} {e} {a} {q}')
                    lists.append(f'{e} {t} {a} {q}')

    return lists

def key_to_query(time, entity, account):
    lists = []
    time_ = time + 1
    q_prev = f"SELECT frmtrm_amount FROM receipts WHERE account_nm = '{account}' AND bsns_year = {time_}"
    q_this = f"SELECT thstrm_amount FROM receipts WHERE account_nm = '{account}' AND bsns_year = {time}"
    lists.append(q_prev)
    lists.append(q_this)
    return lists

def query_to_answer(query):
    ans = list(db.query(query).all()[0].as_dict().values())[0]
    # list(db.query("SELECT thstrm_amount FROM receipts WHERE account_nm = '유동자산' AND bsns_year = 2020").all()[0].as_dict().values())[0]
    # ans = db.query(query).all()[0].as_dict()['thstrm_amount'] 
    return ans


accounts = ['유동자산', '비유동자산', '자산총계', '유동부채', '비유동부채', '부채총계', '이익잉여금', '자본총계', '매출액', '영업이익', '법인세차감전 순이익', '당기순이익', '자본금']
times = np.arange(2015, 2021, 1).tolist()
entities = ['삼성전자']

nlsqls = []

for t in times:
    for e in entities:
        for a in accounts:
            nls   = key_to_nl(t, e, a)
            querys = key_to_query(t, e, a)
            # print('querys: ',querys)
            for query in querys:
                try:
                    ans   = str(query_to_answer(query)) 
                    # print('haha')

                except:
                    # print(query)
                    ans   = False
                    
                    pass

                for nl in nls:
                    if ans:
                        s = [nl, query, ans]
                        # print("\t".join(s) + "\n") 
                        nlsqls.append(s)
                    else:
                        # print(query)
                        pass

import csv 
with open('NLSQL.tsv', 'wt') as f:
    writer = csv.writer(f, delimiter='\t')
    for nlsql in nlsqls:
        writer.writerow(nlsql) 