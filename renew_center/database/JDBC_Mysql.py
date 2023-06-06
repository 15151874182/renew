import jaydebeapi
from DatabaseTools.Mysql import Mysql
from center.tools.logger_function import get_logger

logger = get_logger("DataProcessor")


class JDBC_Mysql_kb(Mysql):
    '''
    jdbc连接金仓数据库
    '''

    def __init__(self, database):

        try:
            driver = database.get_para("driver")
            database_user = database.get_para("database_user")
            database_passwd = database.get_para("database_passwd")
            database_port = 5236 if "port" not in database.config_info.keys() else database.get_para(
                "port")
            database_name = None if "database_name" not in database.config_info.keys() else database.get_para(
                "database_name")
            database_urls = database.get_para("database_url").split()

            jar_file = database.get_para("jar_file")
        except:
            driver = database["driver"]
            database_user = database["database_user"]
            database_passwd = database["database_passwd"]
            database_port = 5236 if "port" not in database.keys(
            ) else database["port"]
            database_name = None if "database_name" not in database.keys(
            ) else database["database_name"]
            database_urls = database["database_url"].split()

            jar_file = database["jar_file"]

        # port = ":" + port
        self.conn = None

        for url in database_urls:
            database_url_str = f"jdbc:kingbase://{url}{database_port}/{database_name}"
            logger.info(f"jdbc connect url: {database_url_str}")
            try:

                self.conn = jaydebeapi.connect(
                    driver, database_url_str, [
                        database_user, database_passwd], jar_file
                )
                self.cursor = self.conn.cursor()
                break
            except:
                logger.info(f"jdbc connect url: {database_url_str} failed!")
        if self.conn is None:
            logger.warning("none of database_urls are useable")


class JDBC_sql(Mysql):
    '''
    jdbc连接达梦数据库
    '''

    def __init__(self, database):

        try:
            driver = database.get_para("driver")
            database_user = database.get_para("database_user")
            database_passwd = database.get_para("database_passwd")
            # database_port = 5236 if "port" not in database.config_info.keys() else database.get_para(
            #     "port")
            # database_name = None if "database_name" not in database.config_info.keys() else database.get_para(
            #     "database_name")
            database_urls = database.get_para("database_url").split()

            jar_file = database.get_para("jar_file")
        except:
            driver = database["driver"]
            database_user = database["database_user"]
            database_passwd = database["database_passwd"]
            # database_port = 5236 if "port" not in database.keys() else database["port"]
            # database_name = None if "database_name" not in database.keys() else database["database_name"]
            database_urls = database["database_url"].split()

            jar_file = database["jar_file"]

        for url in database_urls:
            database_url_str = "jdbc:dm://" + url
            logger.info(f"jdbc connect url: {database_url_str}")
            try:

                self.conn = jaydebeapi.connect(
                    driver, database_url_str, [
                        database_user, database_passwd], jar_file
                )
                self.cursor = self.conn.cursor()
                break
            except:
                logger.info(f"jdbc connect url: {database_url_str} failed!")

        if self.conn is None:
            logger.warning("none of database_urls are useable")
