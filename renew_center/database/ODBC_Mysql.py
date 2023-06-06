import pyodbc
import os
from center.DatabaseTools.Mysql import Mysql
import traceback
from center.tools.logger_function import get_logger

logger = get_logger("DataProcessor")


class ODBC_Mysql(Mysql):
    '''
    odbc连接数据库
    '''

    def __init__(self, database_config):

        try:
            driver = database_config.get_para("driver_odbc")
            database_user = database_config.get_para("database_user")
            database_passwd = database_config.get_para("database_passwd")
            database_port = 5236 if "port" not in database_config.config_info.keys() else database_config.get_para(
                "port")
            database_name = None if "database_name" not in database_config.config_info.keys() else database_config.get_para(
                "database_name")
            database_urls = database_config.get_para("database_url").split()
        except:
            driver = database_config["driver_odbc"]
            database_user = database_config["database_user"]
            database_passwd = database_config["database_passwd"]
            database_port = 5236 if "port" not in database_config.keys(
            ) else database_config["port"]
            database_name = None if "database_name" not in database_config.keys(
            ) else database_config["database_name"]
            database_urls = database_config["database_url"].split()
            traceback.print_exc()

        # 对多个数据库url的处理
        conns = []  
        conn = None
        for database_url in database_urls:

            conn_str = f"DRIVER={{{driver}}};SERVER={database_url};PORT={database_port};" \
                f"UID={database_user};PWD={database_passwd}"
            if database_name is not None:
                conn_str += f";DATABASE={database_name}"
            print(conn_str)
            try:
                conn = pyodbc.connect(conn_str)
                logger.info(f"this url {database_url} connect success")
            except:
                logger.info(f"this url {database_url} connect fail")
                continue
            conns.append(conn)

        if not len(conns):
            logger.warning("none of database_urls are useable")
            assert 0
        self.conn = conns[0]
        self.cursor = self.conn.cursor()
