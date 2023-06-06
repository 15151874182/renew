from ftplib import FTP

class FTPConnect():
    def __init__(self, ftp_config):
        host = ftp_config["host"]
        port = ftp_config["port"]
        username = ftp_config["username"]
        password = ftp_config["password"]

        self.ftp = FTP()
        # ftp.set_debuglevel(2)         # 打开调试级别2，显示详细信息
        self.ftp.connect(host, port)  # 连接
        self.ftp.login(username, password)  # 登录，如果匿名登录则用空串代替即可

    def download(self):
        pass
