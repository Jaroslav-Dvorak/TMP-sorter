from threading import Thread
from time import sleep
import platform    # For getting the operating system name
import subprocess  # For executing a shell command

from instances import movecontrol, gui
from Server import server


# TODO
# - dodělat diagnostiku


def ping(host):
    """
    Returns True if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
    """

    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower() == 'windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, '1', host]

    return subprocess.call(command) == 0


while not ping("10.83.1.2"):
    print("čekejte prosím...")
    sleep(1)

if __name__ == "__main__":
    t_server = Thread(target=server)
    t_server.start()
    t_movecontrol = Thread(target=movecontrol.run)
    t_movecontrol.start()
    gui()
