import os
from pyftpdlib.servers import FTPServer
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from ImageRecognizer import recognizer, gui_computer
from MoveControl import movecontrol
from is_rpi import is_rpi


def server():
    authorizer = DummyAuthorizer()
    homedir = "/mnt/ramdisk" if is_rpi() else './FTP'
    homedir = "/mnt/ramdisk" if is_rpi() else '/Volumes/RAM_drive/'
    authorizer.add_user('IV', '12345', homedir=homedir, perm='elradfmwMT')
    handler = MyHandler
    handler.authorizer = authorizer
    the_server = FTPServer(('', 21), handler)
    the_server.serve_forever()


class MyHandler(FTPHandler):

    def on_connect(self):
        movecontrol.cam_connected = True
        # print("%s:%s connected" % (self.remote_ip, self.remote_port))
        pass

    def on_disconnect(self):
        movecontrol.cam_connected = False
        pass

    def on_login(self, username):
        pass

    def on_logout(self, username):
        # do something when user logs out
        pass

    def on_file_sent(self, file):
        # do something when a file has been sent
        pass

    def on_file_received(self, file):
        ok = recognizer.evaluator(file)
        movecontrol.badone = not ok
        gui_computer.load_imge(file)
        os.remove(file)
        pass

    def on_incomplete_file_sent(self, file):
        # do something when a file is partially sent
        pass

    def on_incomplete_file_received(self, file):
        # remove partially uploaded files
        os.remove(file)
