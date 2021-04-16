
from impala.dbapi import connect
from krbcontext import krbcontext

import pandas as pd
import utils.txt_process as tp
import time
from utils.config import project_root_path


def get_data(hostname, port, target_table):
    with krbcontext(using_keytab=True, principal='impala',
                    keytab_file=project_root_path+'/impala.keytab',
                    ccache_file='krb5cc_0'):
        # 2.调用impaly初始化connector并获取游标
        conn = connect(hostname, port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
        cur = conn.cursor()
        sql = "select * from {0}".format(target_table)
        cur.execute(sql)
        # data_list = cur.fetchall()
        data_list = []
        for row in cur:
            data_list.append(row)

        return data_list


def get_data_sql(hostname, port, sql):
    with krbcontext(using_keytab=True, principal='impala',
                    keytab_file=project_root_path+'/impala.keytab',
                    ccache_file='krb5cc_0'):
        # 2.调用impaly初始化connector并获取游标
        conn = connect(hostname, port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
        cur = conn.cursor()
        cur.execute(sql)
        # data_list = cur.fetchall()
        data_list = []
        for row in cur:
            data_list.append(row)

        return data_list

def get_data_sql_with_columns(hostname, port, sql):
    def parse_des(desc):
        cols = [i[0] for i in desc]
        return cols

    with krbcontext(using_keytab=True, principal='impala',
                    keytab_file='/home/keytab/impala.keytab',
                    ccache_file='krb5cc_0'):
        # 2.调用impaly初始化connector并获取游标
        conn = connect(hostname, port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
        cur = conn.cursor()
        cur.execute(sql)
        data_list = cur.fetchall()
        cols = parse_des(cur.description)
        data_list = pd.DataFrame(data_list,columns=cols)
        return data_list


def get_data_frame_sql(hostname, port, sql, columns):
    data_list = get_data_sql(hostname, port, sql)
    df = pd.DataFrame(data_list, columns=columns)
    return df
    # for key, value in data.items():
    #    pipeline.set(key, value)
    # pipeline.execute()


def execute_sql(hostname, port, sql_path):
    with krbcontext(using_keytab=True, principal='impala',
                    keytab_file='/home/keytab/impala.keytab',
                    ccache_file='krb5cc_0'):
        # 2.调用impaly初始化connector并获取游标
        conn = connect(hostname, port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
        cur = conn.cursor()
        sql_context = tp.read_sql(sql_path)
        sql_commands = sql_context.split(';')

        for command in sql_commands[:-1]:
            # print(command)
            start_at = time.time()
            cur.execute(command)
            time.sleep(5)
            print(cur.query_string)
            end_at = time.time()
            print('Command execute complete! Execute duration seconds :{}'.format(end_at - start_at))


class Impala:
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port

    def execute_sql(self, sql_path):
        return execute_sql(self.hostname, self.port, sql_path)

    def get_data_frame_sql(self, sql, columns):
        return get_data_frame_sql(self.hostname, self.port, sql, columns)
