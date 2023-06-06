#!/usr/bin/python3


import pandas as pd
from datetime import datetime, date
import numpy as np

from center.tools.logger_function import get_logger

logger = get_logger("DataProcessor")


class Mysql():
    def __init__(self, database: dict):
        import pymysql
        self.conn = pymysql.connect(host=database['host'],
                                    port=database['port'],
                                    db=database['db'],
                                    user=database['user'],
                                    passwd=database['passwd'],
                                    cursorclass=pymysql.cursors.DictCursor
                                    # charset=database['charset']
                                    )
        self.cursor = self.conn.cursor()

    def __del__(self):
        # 析构函数
        self.cursor.close()
        self.conn.close()

    def execute(self, sql, col_names=None):
        '''
        执行函数
        :param sql:
        :param col_name:
        :return:
        '''
        logger.info(f"execute sql:\n\t{sql}")
        if col_names is None:
            pd_data = pd.read_sql(sql, self.conn)
            return pd_data
        try:
            self.cursor.execute(sql)
        except:
            logger.warning(f"run sql: {sql} Failed!!!!!")
        # cursor.fetchall() 和 np.array 最好分开，出问题容易检查
        fetch_data = self.cursor.fetchall()
        data = np.array(fetch_data)
        self.conn.commit()
        if not len(data):
            logger.warning(f"{sql} get NULL Data")
            return pd.DataFrame([], columns=col_names)
        pd_data = pd.DataFrame(data, columns=col_names)

        return pd_data

    def insert(self, table_name, df_data, primary_key=["date", "id"]):
        """
        插入数据到数据库
        :param table_name:  表名
        :param df_data:     DataFrame，插入的数据，要求按照数据库表的列名来
        :param primary_key: 主键，根据主键删除数据，保证插入数据的唯一性
        :return:
        """
        col_list = list(df_data.columns)
        sql_head = f'INSERT INTO {table_name}({",".join(col_list)}) '
        delete_head = f'delete from {table_name} '

        for i in df_data.index:
            # delete_sql = delete_head + \
            #              f"where {time_name}='{df_data.loc[i, time_name]}' and {id_name}={df_data.loc[i, id_name]};"
            # print(f"df_data is {df_data}")
            if len(primary_key):
                delete_sql = delete_head + "where "
                for key in primary_key:
                    delete_sql = delete_sql + \
                        f"{key}='{df_data.loc[i, key]}' and "
                delete_sql = delete_sql[:-5] + ";"
            else:
                delete_sql = delete_head + ";"

            try:
                self.cursor.execute(delete_sql)
            except:
                logger.warning(f"execute delete sql:{delete_sql} failed!")

            # sql_value = f"VALUES({', '.join(df_data.loc[i, :].tolist())});"
            sql_value = "VALUES("
            values = df_data.loc[i, :].tolist()
            for v in values:
                if type(v) in [str, pd._libs.tslibs.timestamps.Timestamp, datetime]:
                    sql_value += f"'{v}', "
                elif v is None or np.isnan(v):
                    sql_value += "NULL, "
                else:
                    sql_value += f"{v}, "
            sql_value = sql_value[:-2] + ");"

            sql_i = sql_head + sql_value

            logger.debug(f"execute insert sql:\n{sql_i}.")
            try:
                self.cursor.execute(sql_i)
            except:
                logger.warning(f"execute insert sql:\n{sql_i} failed!")
            self.conn.commit()

    def insert_before(self, table_name, df_data, time_name="date", id_name="id"):
        """
        原先插入的接口
        :param table_name:
        :param df_data:
        :param time_name:
        :param id_name:
        :return:
        """
        col_list = list(df_data.columns)
        sql_head = f'INSERT INTO {table_name}({",".join(col_list)}) '
        delete_head = f'delete from {table_name} '

        for i in df_data.index:
            # delete_sql = delete_head + \
            #              f"where {time_name}='{df_data.loc[i, time_name]}' and {id_name}={df_data.loc[i, id_name]};"
            # print(f"df_data is {df_data}")
            delete_sql = delete_head + \
                f"where {id_name}='{df_data.loc[i, id_name]}'"
            if len(time_name):
                delete_sql += f" and {time_name}='{df_data.loc[i, time_name]}';"
            else:
                delete_sql += ";"

            try:
                self.cursor.execute(delete_sql)
            except:
                logger.warning(f"execute delete sql:{delete_sql} failed!")

            # sql_value = f"VALUES({', '.join(df_data.loc[i, :].tolist())});"
            sql_value = "VALUES("
            values = df_data.loc[i, :].tolist()
            for v in values:
                if type(v) in [str, pd._libs.tslibs.timestamps.Timestamp, datetime]:
                    sql_value += f"'{v}', "
                elif v is None or np.isnan(v):
                    sql_value += "NULL, "
                else:
                    sql_value += f"{v}, "
            sql_value = sql_value[:-2] + ");"

            sql_i = sql_head + sql_value

            logger.debug(f"execute insert sql:\n{sql_i}.")
            try:
                self.cursor.execute(sql_i)
            except:
                logger.warning(f"execute insert sql:\n{sql_i} failed!")
            self.conn.commit()

    def get_table(self, table_name, col_names=None, **kwargs):
        '''
        fetch table函数
        :param table_name:  表名
        :param col_names:   输入列名，若不输入，则默认获取所有列
        :param kwargs:      读表的其他参数，例如 data_type='相对湿度'
        :return:
        '''
        if col_names is None:
            sql = f"select * from {table_name} "
        else:
            col_names_str = ", ".join(col_names)
            sql = f"select {col_names_str} from {table_name} "

        if not len(kwargs.keys()):
            sql += ";"
            data = self.execute(sql, col_names)
            return data

        sql += "where "
        for k in kwargs.keys():
            if type(kwargs[k]) in [str, pd._libs.tslibs.timestamps.Timestamp, date, datetime]:
                sql += f"{k}='{kwargs[k]}' and "
            else:
                sql += f"{k}={kwargs[k]} and "
        sql = sql[:-5] + ";"

        data = self.execute(sql, col_names)
        return data

    def get_table_time(self, table_name, col_names=None,
                       time_name="UPDATE_TIME", start_time="", end_time="",
                       **kwargs):
        '''
        :param table_name:  表名
        :param col_name:    表的列名，如不输入则获取所有列，且返回DataFrame列名为空
        :param time_name:   时间列名，如不输入则不会根据时间排序
        :param start_time:  选择数据的起始时间
        :param end_time:    选择数据的终止时间
        :param kwargs:      其他入参，可以id = [1,2]或者id=3
        :return:
        '''
        # 写where的sql语句
        where_sql = ""
        if len(kwargs):
            where_sql = " where "
            for k in kwargs.keys():
                values = kwargs[k]
                if type(values) == list:
                    if type(values[0]) == str:
                        values_sql = "'" + "','".join(values) + "'"
                    else:
                        values_sql = ""
                        for value in values:
                            values_sql += f"{value}, "
                        values_sql = values_sql[:-2]
                    where_sql += f"{k} in ({values_sql}) and "
                    # for value in values:
                    #     where_sql += f"{k}={value} or  "
                elif type(values) == str:
                    where_sql += f"{k}='{values}' and "
                else:
                    where_sql += f"{k}={values} and "
            where_sql = where_sql[:-5]

        if len(start_time) or len(end_time):
            if not len(where_sql):
                where_sql = " where "
            else:
                where_sql += " and "

            if len(start_time):
                where_sql += f"{time_name}>='{start_time}' and "
            if len(end_time):
                where_sql += f"{time_name}<='{end_time}' and "
            where_sql = where_sql[:-5]

        end_sql = f" order by {time_name} asc;" if len(time_name) else ";"
        # desc

        if col_names is not None:
            separation_sign = ","
            col_list = separation_sign.join(col_names)

            sql = f"select {col_list} from {table_name}" + where_sql + end_sql
        else:
            sql = (f"select * from {table_name} {where_sql} {end_sql};")

        logger.info(sql)
        data = self.execute(sql, col_names)

        # 将数据保存到指定的路径
        return data


if __name__ == '__main__':
    try:
        from Mysql_config import weather_config
    except:
        from DatabaseTools.Mysql_config import weather_config

    a = Mysql(weather_config)
    # sql = "SELECT * from load_table limit 20;"
    # print(a.execute(sql))
    # table = a.get_table("load_table", area='diaodu')
    # table.to_csv("diaodu_load.csv", index = False, encoding="utf_8_sig")
    # print(table)

    table = a.get_table("weather_table", area='diaodu')
    table.to_csv("diaodu_weather.csv", index=False, encoding="utf_8_sig")
    print(table)
