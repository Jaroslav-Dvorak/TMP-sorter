from threading import Thread
from GUI.StartGUI import StartGUI
from MoveControl import movecontrol
from Server import server


if __name__ == "__main__":
    t_server = Thread(target=server)
    t_server.start()
    t_movecontrol = Thread(target=movecontrol.run)
    t_movecontrol.start()
    gui = StartGUI()
